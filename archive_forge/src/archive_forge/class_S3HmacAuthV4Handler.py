import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
class S3HmacAuthV4Handler(HmacAuthV4Handler, AuthHandler):
    """
    Implements a variant of Version 4 HMAC authorization specific to S3.
    """
    capability = ['hmac-v4-s3']

    def __init__(self, *args, **kwargs):
        super(S3HmacAuthV4Handler, self).__init__(*args, **kwargs)
        if self.region_name:
            self.region_name = self.clean_region_name(self.region_name)

    def clean_region_name(self, region_name):
        if region_name.startswith('s3-'):
            return region_name[3:]
        return region_name

    def canonical_uri(self, http_request):
        path = urllib.parse.urlparse(http_request.path)
        path_str = six.ensure_str(path.path)
        unquoted = urllib.parse.unquote(path_str)
        encoded = urllib.parse.quote(unquoted, safe='/~')
        return encoded

    def canonical_query_string(self, http_request):
        l = []
        for param in sorted(http_request.params):
            value = get_utf8able_str(http_request.params[param])
            l.append('%s=%s' % (urllib.parse.quote(param, safe='-_.~'), urllib.parse.quote(value, safe='-_.~')))
        return '&'.join(l)

    def host_header(self, host, http_request):
        port = http_request.port
        secure = http_request.protocol == 'https'
        if port == 80 and (not secure) or (port == 443 and secure):
            return http_request.host
        return '%s:%s' % (http_request.host, port)

    def headers_to_sign(self, http_request):
        """
        Select the headers from the request that need to be included
        in the StringToSign.
        """
        host_header_value = self.host_header(self.host, http_request)
        headers_to_sign = {'Host': host_header_value}
        for name, value in http_request.headers.items():
            lname = name.lower()
            if lname not in ['authorization']:
                headers_to_sign[name] = value
        return headers_to_sign

    def determine_region_name(self, host):
        parts = self.split_host_parts(host)
        if self.region_name is not None:
            region_name = self.region_name
        elif len(parts) == 3:
            region_name = self.clean_region_name(parts[0])
            if region_name == 's3':
                region_name = 'us-east-1'
        else:
            for offset, part in enumerate(reversed(parts)):
                part = part.lower()
                if part == 's3':
                    region_name = parts[-offset]
                    if region_name == 'amazonaws':
                        region_name = 'us-east-1'
                    break
                elif part.startswith('s3-'):
                    region_name = self.clean_region_name(part)
                    break
        return region_name

    def determine_service_name(self, host):
        return 's3'

    def mangle_path_and_params(self, req):
        """
        Returns a copy of the request object with fixed ``auth_path/params``
        attributes from the original.
        """
        modified_req = copy.copy(req)
        parsed_path = urllib.parse.urlparse(modified_req.auth_path)
        modified_req.auth_path = parsed_path.path
        if modified_req.params is None:
            modified_req.params = {}
        else:
            copy_params = req.params.copy()
            modified_req.params = copy_params
        raw_qs = parsed_path.query
        existing_qs = parse_qs_safe(raw_qs, keep_blank_values=True)
        for key, value in existing_qs.items():
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    existing_qs[key] = value[0]
        modified_req.params.update(existing_qs)
        return modified_req

    def payload(self, http_request):
        if http_request.headers.get('x-amz-content-sha256'):
            return http_request.headers['x-amz-content-sha256']
        return super(S3HmacAuthV4Handler, self).payload(http_request)

    def add_auth(self, req, **kwargs):
        if 'x-amz-content-sha256' not in req.headers:
            if '_sha256' in req.headers:
                req.headers['x-amz-content-sha256'] = req.headers.pop('_sha256')
            else:
                req.headers['x-amz-content-sha256'] = self.payload(req)
        updated_req = self.mangle_path_and_params(req)
        return super(S3HmacAuthV4Handler, self).add_auth(updated_req, unmangled_req=req, **kwargs)

    def presign(self, req, expires, iso_date=None):
        """
        Presign a request using SigV4 query params. Takes in an HTTP request
        and an expiration time in seconds and returns a URL.

        http://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html
        """
        if iso_date is None:
            iso_date = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        region = self.determine_region_name(req.host)
        service = self.determine_service_name(req.host)
        params = {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': '%s/%s/%s/%s/aws4_request' % (self._provider.access_key, iso_date[:8], region, service), 'X-Amz-Date': iso_date, 'X-Amz-Expires': expires, 'X-Amz-SignedHeaders': 'host'}
        if self._provider.security_token:
            params['X-Amz-Security-Token'] = self._provider.security_token
        headers_to_sign = self.headers_to_sign(req)
        l = sorted(['%s' % n.lower().strip() for n in headers_to_sign])
        params['X-Amz-SignedHeaders'] = ';'.join(l)
        req.params.update(params)
        cr = self.canonical_request(req)
        cr = '\n'.join(cr.split('\n')[:-1]) + '\nUNSIGNED-PAYLOAD'
        req.headers['X-Amz-Date'] = iso_date
        sts = self.string_to_sign(req, cr)
        signature = self.signature(req, sts)
        req.params['X-Amz-Signature'] = signature
        return '%s://%s%s?%s' % (req.protocol, req.host, req.path, urllib.parse.urlencode(req.params))