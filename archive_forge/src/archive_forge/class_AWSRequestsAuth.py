import hmac
import hashlib
import datetime
import requests
class AWSRequestsAuth(requests.auth.AuthBase):
    """
    Auth class that allows us to connect to AWS services
    via Amazon's signature version 4 signing process

    Adapted from https://docs.aws.amazon.com/general/latest/gr/sigv4-signed-request-examples.html
    """

    def __init__(self, aws_access_key, aws_secret_access_key, aws_host, aws_region, aws_service, aws_token=None):
        """
        Example usage for talking to an AWS Elasticsearch Service:

        AWSRequestsAuth(aws_access_key='YOURKEY',
                        aws_secret_access_key='YOURSECRET',
                        aws_host='search-service-foobar.us-east-1.es.amazonaws.com',
                        aws_region='us-east-1',
                        aws_service='es',
                        aws_token='...')

        The aws_token is optional and is used only if you are using STS
        temporary credentials.
        """
        self.aws_access_key = aws_access_key
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_host = aws_host
        self.aws_region = aws_region
        self.service = aws_service
        self.aws_token = aws_token

    def __call__(self, r):
        """
        Adds the authorization headers required by Amazon's signature
        version 4 signing process to the request.

        Adapted from https://docs.aws.amazon.com/general/latest/gr/sigv4-signed-request-examples.html
        """
        aws_headers = self.get_aws_request_headers_handler(r)
        r.headers.update(aws_headers)
        return r

    def get_aws_request_headers_handler(self, r):
        """
        Override get_aws_request_headers_handler() if you have a
        subclass that needs to call get_aws_request_headers() with
        an arbitrary set of AWS credentials. The default implementation
        calls get_aws_request_headers() with self.aws_access_key,
        self.aws_secret_access_key, and self.aws_token
        """
        return self.get_aws_request_headers(r=r, aws_access_key=self.aws_access_key, aws_secret_access_key=self.aws_secret_access_key, aws_token=self.aws_token)

    def get_aws_request_headers(self, r, aws_access_key, aws_secret_access_key, aws_token):
        """
        Returns a dictionary containing the necessary headers for Amazon's
        signature version 4 signing process. An example return value might
        look like

            {
                'Authorization': 'AWS4-HMAC-SHA256 Credential=YOURKEY/20160618/us-east-1/es/aws4_request, '
                                 'SignedHeaders=host;x-amz-date, '
                                 'Signature=ca0a856286efce2a4bd96a978ca6c8966057e53184776c0685169d08abd74739',
                'x-amz-date': '20160618T220405Z',
            }
        """
        t = datetime.datetime.utcnow()
        amzdate = t.strftime('%Y%m%dT%H%M%SZ')
        datestamp = t.strftime('%Y%m%d')
        canonical_uri = AWSRequestsAuth.get_canonical_path(r)
        canonical_querystring = AWSRequestsAuth.get_canonical_querystring(r)
        canonical_headers = 'host:' + self.aws_host + '\n' + 'x-amz-date:' + amzdate + '\n'
        if aws_token:
            canonical_headers += 'x-amz-security-token:' + aws_token + '\n'
        signed_headers = 'host;x-amz-date'
        if aws_token:
            signed_headers += ';x-amz-security-token'
        body = r.body if r.body else bytes()
        try:
            body = body.encode('utf-8')
        except (AttributeError, UnicodeDecodeError):
            body = body
        payload_hash = hashlib.sha256(body).hexdigest()
        canonical_request = r.method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = datestamp + '/' + self.aws_region + '/' + self.service + '/' + 'aws4_request'
        string_to_sign = algorithm + '\n' + amzdate + '\n' + credential_scope + '\n' + hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        signing_key = getSignatureKey(aws_secret_access_key, datestamp, self.aws_region, self.service)
        string_to_sign_utf8 = string_to_sign.encode('utf-8')
        signature = hmac.new(signing_key, string_to_sign_utf8, hashlib.sha256).hexdigest()
        authorization_header = algorithm + ' ' + 'Credential=' + aws_access_key + '/' + credential_scope + ', ' + 'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature
        headers = {'Authorization': authorization_header, 'x-amz-date': amzdate, 'x-amz-content-sha256': payload_hash}
        if aws_token:
            headers['X-Amz-Security-Token'] = aws_token
        return headers

    @classmethod
    def get_canonical_path(cls, r):
        """
        Create canonical URI--the part of the URI from domain to query
        string (use '/' if no path)
        """
        parsedurl = urlparse(r.url)
        return quote(parsedurl.path if parsedurl.path else '/', safe='/-_.~')

    @classmethod
    def get_canonical_querystring(cls, r):
        """
        Create the canonical query string. According to AWS, by the
        end of this function our query string values must
        be URL-encoded (space=%20) and the parameters must be sorted
        by name.

        This method assumes that the query params in `r` are *already*
        url encoded.  If they are not url encoded by the time they make
        it to this function, AWS may complain that the signature for your
        request is incorrect.

        It appears elasticsearc-py url encodes query paramaters on its own:
            https://github.com/elastic/elasticsearch-py/blob/5dfd6985e5d32ea353d2b37d01c2521b2089ac2b/elasticsearch/connection/http_requests.py#L64

        If you are using a different client than elasticsearch-py, it
        will be your responsibility to urleconde your query params before
        this method is called.
        """
        canonical_querystring = ''
        parsedurl = urlparse(r.url)
        querystring_sorted = '&'.join(sorted(parsedurl.query.split('&')))
        for query_param in querystring_sorted.split('&'):
            key_val_split = query_param.split('=', 1)
            key = key_val_split[0]
            if len(key_val_split) > 1:
                val = key_val_split[1]
            else:
                val = ''
            if key:
                if canonical_querystring:
                    canonical_querystring += '&'
                canonical_querystring += u'='.join([key, val])
        return canonical_querystring