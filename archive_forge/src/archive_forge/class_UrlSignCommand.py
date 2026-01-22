from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
import copy
from datetime import datetime
from datetime import timedelta
import getpass
import json
import re
import sys
import six
from six.moves import urllib
from apitools.base.py.exceptions import HttpError
from apitools.base.py.http_wrapper import MakeRequest
from apitools.base.py.http_wrapper import Request
from boto import config
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils import constants
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.shim_util import GcloudStorageMap, GcloudStorageFlag
from gslib.utils.signurl_helper import CreatePayload, GetFinalUrl
class UrlSignCommand(Command):
    """Implementation of gsutil url_sign command."""
    command_spec = Command.CreateCommandSpec('signurl', command_name_aliases=['signedurl', 'queryauth'], usage_synopsis=_SYNOPSIS, min_args=1, max_args=constants.NO_MAX, supported_sub_args='m:d:b:c:p:r:u', supported_private_args=['use-service-account'], file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreFileURLsArgument(), CommandArgument.MakeZeroOrMoreCloudURLsArgument()])
    help_spec = Command.HelpSpec(help_name='signurl', help_name_aliases=['signedurl', 'queryauth'], help_type='command_help', help_one_line_summary='Create a signed URL', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})

    def get_gcloud_storage_args(self):
        original_args = copy.deepcopy(self.args)
        original_sub_opts = copy.deepcopy(self.sub_opts)
        gcloud_command = ['storage', 'sign-url', '--format=csv[separator="\\t"](resource:label="URL", http_verb:label="HTTP Method", expiration:label="Expiration", signed_url:label="Signed URL")', '--private-key-file=' + self.args[0]]
        self.args = self.args[1:]
        duration_arg_idx = None
        http_verb_arg_idx = None
        content_type_arg_idx = None
        billing_project_arg_idx = None
        for i, (flag, _) in enumerate(self.sub_opts):
            if flag == '-d':
                duration_arg_idx = i
            elif flag == '-m':
                http_verb_arg_idx = i
            elif flag == '-c':
                content_type_arg_idx = i
            elif flag == '-b':
                billing_project_arg_idx = i
        if duration_arg_idx is not None:
            seconds = str(int(_DurationToTimeDelta(self.sub_opts[duration_arg_idx][1]).total_seconds())) + 's'
            self.sub_opts[duration_arg_idx] = ('-d', seconds)
        if http_verb_arg_idx is not None:
            if self.sub_opts[http_verb_arg_idx][1] == 'RESUMABLE':
                self.sub_opts[http_verb_arg_idx] = ('-m', 'POST')
                gcloud_command += ['--headers=x-goog-resumable=start']
        if content_type_arg_idx is not None:
            content_type_value = self.sub_opts[content_type_arg_idx][1]
            self.sub_opts[content_type_arg_idx] = ('-c', 'content-type=' + content_type_value)
        if billing_project_arg_idx is not None:
            project_value = self.sub_opts[billing_project_arg_idx][1]
            self.sub_opts[billing_project_arg_idx] = ('-b', 'userProject=' + project_value)
        fully_translated_command = super().get_gcloud_storage_args(GcloudStorageMap(gcloud_command=gcloud_command, flag_map={'-m': GcloudStorageFlag('--http-verb'), '-d': GcloudStorageFlag('--duration'), '-b': GcloudStorageFlag('--query-params'), '-c': GcloudStorageFlag('--headers'), '-r': GcloudStorageFlag('--region'), '-p': GcloudStorageFlag('--private-key-password')}))
        self.args = original_args
        self.sub_opts = original_sub_opts
        return fully_translated_command

    def _ParseAndCheckSubOpts(self):
        delta = None
        method = 'GET'
        content_type = ''
        passwd = None
        region = _AUTO_DETECT_REGION
        use_service_account = False
        billing_project = None
        for o, v in self.sub_opts:
            if six.PY2:
                v = v.decode(sys.stdin.encoding or constants.UTF8)
            if o == '-d':
                if delta is not None:
                    delta += _DurationToTimeDelta(v)
                else:
                    delta = _DurationToTimeDelta(v)
            elif o == '-m':
                method = v
            elif o == '-c':
                content_type = v
            elif o == '-p':
                passwd = v
            elif o == '-r':
                region = v
            elif o == '-u' or o == '--use-service-account':
                use_service_account = True
            elif o == '-b':
                billing_project = v
            else:
                self.RaiseInvalidArgumentException()
        if delta is None:
            delta = timedelta(hours=1)
        elif use_service_account and delta > _MAX_EXPIRATION_TIME_WITH_MINUS_U:
            raise CommandException('Max valid duration allowed is %s when -u flag is used. For longer duration, consider using the private-key-file instead of the -u option.' % _MAX_EXPIRATION_TIME_WITH_MINUS_U)
        elif delta > _MAX_EXPIRATION_TIME:
            raise CommandException('Max valid duration allowed is %s' % _MAX_EXPIRATION_TIME)
        if method not in ['GET', 'PUT', 'DELETE', 'HEAD', 'RESUMABLE']:
            raise CommandException('HTTP method must be one of[GET|HEAD|PUT|DELETE|RESUMABLE]')
        if not use_service_account and len(self.args) < 2:
            raise CommandException('The command requires a key file argument and one or more URL arguments if the --use-service-account flag is missing. Run `gsutil help signurl` for more info')
        if use_service_account and billing_project:
            raise CommandException('Specifying both the -b and --use-service-account options together isinvalid.')
        return (method, delta, content_type, passwd, region, use_service_account, billing_project)

    def _ProbeObjectAccessWithClient(self, key, use_service_account, provider, client_email, gcs_path, generation, logger, region, billing_project):
        """Performs a head request against a signed URL to check for read access."""
        signed_url = _GenSignedUrl(key=key, api=self.gsutil_api, use_service_account=use_service_account, provider=provider, client_id=client_email, method='HEAD', duration=timedelta(seconds=60), gcs_path=gcs_path, generation=generation, logger=logger, region=region, billing_project=billing_project, string_to_sign_debug=True)
        try:
            h = GetNewHttp()
            req = Request(signed_url, 'HEAD')
            response = MakeRequest(h, req)
            if response.status_code not in [200, 403, 404]:
                raise HttpError.FromResponse(response)
            return response.status_code
        except HttpError as http_error:
            if http_error.has_attr('response'):
                error_response = http_error.response
                error_string = 'Unexpected HTTP response code %s while querying object readability. Is your system clock accurate?' % error_response.status_code
                if error_response.content:
                    error_string += ' Content: %s' % error_response.content
            else:
                error_string = 'Expected an HTTP response code of 200 while querying object readability, but received an error: %s' % http_error
            raise CommandException(error_string)

    def _EnumerateStorageUrls(self, in_urls):
        ret = []
        for url_str in in_urls:
            if ContainsWildcard(url_str):
                ret.extend([blr.storage_url for blr in self.WildcardIterator(url_str)])
            else:
                ret.append(StorageUrlFromString(url_str))
        return ret

    def RunCommand(self):
        """Command entry point for signurl command."""
        if not HAVE_OPENSSL:
            raise CommandException('The signurl command requires the pyopenssl library (try pip install pyopenssl or easy_install pyopenssl)')
        method, delta, content_type, passwd, region, use_service_account, billing_project = self._ParseAndCheckSubOpts()
        arg_start_index = 0 if use_service_account else 1
        storage_urls = self._EnumerateStorageUrls(self.args[arg_start_index:])
        region_cache = {}
        key = None
        if not use_service_account:
            try:
                key, client_email = _ReadJSONKeystore(open(self.args[0], 'rb').read(), passwd)
            except ValueError:
                if not passwd:
                    passwd = getpass.getpass('Keystore password:')
                try:
                    key, client_email = _ReadKeystore(open(self.args[0], 'rb').read(), passwd)
                except ValueError:
                    raise CommandException('Unable to parse private key from {0}'.format(self.args[0]))
        else:
            client_email = self.gsutil_api.GetServiceAccountId(provider='gs')
        print('URL\tHTTP Method\tExpiration\tSigned URL')
        for url in storage_urls:
            if url.scheme != 'gs':
                raise CommandException('Can only create signed urls from gs:// urls')
            if url.IsBucket():
                if region == _AUTO_DETECT_REGION:
                    raise CommandException("Generating signed URLs for creating buckets requires a region be specified via the -r option. Run `gsutil help signurl` for more information about the '-r' option.")
                gcs_path = url.bucket_name
                if method == 'RESUMABLE':
                    raise CommandException('Resumable signed URLs require an object name.')
            else:
                gcs_path = '{0}/{1}'.format(url.bucket_name, urllib.parse.quote(url.object_name.encode(constants.UTF8), safe=b'/~'))
            if region == _AUTO_DETECT_REGION:
                if url.bucket_name in region_cache:
                    bucket_region = region_cache[url.bucket_name]
                else:
                    try:
                        _, bucket = self.GetSingleBucketUrlFromArg('gs://{}'.format(url.bucket_name), bucket_fields=['location'])
                    except Exception as e:
                        raise CommandException("{}: Failed to auto-detect location for bucket '{}'. Please ensure you have storage.buckets.get permission on the bucket or specify the bucket's location using the '-r' option.".format(e.__class__.__name__, url.bucket_name))
                    bucket_region = bucket.location.lower()
                    region_cache[url.bucket_name] = bucket_region
            else:
                bucket_region = region
            final_url = _GenSignedUrl(key=key, api=self.gsutil_api, use_service_account=use_service_account, provider=url.scheme, client_id=client_email, method=method, duration=delta, gcs_path=gcs_path, generation=url.generation, logger=self.logger, region=bucket_region, content_type=content_type, billing_project=billing_project, string_to_sign_debug=True)
            expiration = calendar.timegm((datetime.utcnow() + delta).utctimetuple())
            expiration_dt = datetime.fromtimestamp(expiration)
            time_str = expiration_dt.strftime('%Y-%m-%d %H:%M:%S')
            if six.PY2:
                time_str = time_str.decode(constants.UTF8)
            url_info_str = '{0}\t{1}\t{2}\t{3}'.format(url.url_string, method, time_str, final_url)
            if six.PY2:
                url_info_str = url_info_str.encode(constants.UTF8)
            print(url_info_str)
            response_code = self._ProbeObjectAccessWithClient(key, use_service_account, url.scheme, client_email, gcs_path, url.generation, self.logger, bucket_region, billing_project)
            if response_code == 404:
                if url.IsBucket() and method != 'PUT':
                    raise CommandException('Bucket {0} does not exist. Please create a bucket with that name before a creating signed URL to access it.'.format(url))
                elif method != 'PUT' and method != 'RESUMABLE':
                    raise CommandException('Object {0} does not exist. Please create/upload an object with that name before a creating signed URL to access it.'.format(url))
            elif response_code == 403:
                self.logger.warn('%s does not have permissions on %s, using this link will likely result in a 403 error until at least READ permissions are granted', client_email or 'The account', url)
        return 0