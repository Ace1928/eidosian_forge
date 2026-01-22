from __future__ import absolute_import, division, print_function
import hashlib
import os
import posixpath
import shutil
import io
import tempfile
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.ansible_release import __version__ as ansible_version
from re import match
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
class MavenDownloader:

    def __init__(self, module, base, local=False, headers=None):
        self.module = module
        if base.endswith('/'):
            base = base.rstrip('/')
        self.base = base
        self.local = local
        self.headers = headers
        self.user_agent = 'Ansible {0} maven_artifact'.format(ansible_version)
        self.latest_version_found = None
        self.metadata_file_name = 'maven-metadata-local.xml' if local else 'maven-metadata.xml'

    def find_version_by_spec(self, artifact):
        path = '/%s/%s' % (artifact.path(False), self.metadata_file_name)
        content = self._getContent(self.base + path, 'Failed to retrieve the maven metadata file: ' + path)
        xml = etree.fromstring(content)
        original_versions = xml.xpath('/metadata/versioning/versions/version/text()')
        versions = []
        for version in original_versions:
            try:
                versions.append(Version.coerce(version))
            except ValueError:
                pass
        parse_versions_syntax = {'^\\(,(?P<upper_bound>[0-9.]*)]$': '<={upper_bound}', '^(?P<version>[0-9.]*)$': '~={version}', '^\\[(?P<version>[0-9.]*)\\]$': '=={version}', '^\\[(?P<lower_bound>[0-9.]*),\\s*(?P<upper_bound>[0-9.]*)\\]$': '>={lower_bound},<={upper_bound}', '^\\[(?P<lower_bound>[0-9.]*),\\s*(?P<upper_bound>[0-9.]+)\\)$': '>={lower_bound},<{upper_bound}', '^\\[(?P<lower_bound>[0-9.]*),\\)$': '>={lower_bound}'}
        for regex, spec_format in parse_versions_syntax.items():
            regex_result = match(regex, artifact.version_by_spec)
            if regex_result:
                spec = Spec(spec_format.format(**regex_result.groupdict()))
                selected_version = spec.select(versions)
                if not selected_version:
                    raise ValueError('No version found with this spec version: {0}'.format(artifact.version_by_spec))
                if str(selected_version) not in original_versions:
                    selected_version.patch = None
                return str(selected_version)
        raise ValueError('The spec version {0} is not supported! '.format(artifact.version_by_spec))

    def find_latest_version_available(self, artifact):
        if self.latest_version_found:
            return self.latest_version_found
        path = '/%s/%s' % (artifact.path(False), self.metadata_file_name)
        content = self._getContent(self.base + path, 'Failed to retrieve the maven metadata file: ' + path)
        xml = etree.fromstring(content)
        v = xml.xpath('/metadata/versioning/versions/version[last()]/text()')
        if v:
            self.latest_version_found = v[0]
            return v[0]

    def find_uri_for_artifact(self, artifact):
        if artifact.version_by_spec:
            artifact.version = self.find_version_by_spec(artifact)
        if artifact.version == 'latest':
            artifact.version = self.find_latest_version_available(artifact)
        if artifact.is_snapshot():
            if self.local:
                return self._uri_for_artifact(artifact, artifact.version)
            path = '/%s/%s' % (artifact.path(), self.metadata_file_name)
            content = self._getContent(self.base + path, 'Failed to retrieve the maven metadata file: ' + path)
            xml = etree.fromstring(content)
            for snapshotArtifact in xml.xpath('/metadata/versioning/snapshotVersions/snapshotVersion'):
                classifier = snapshotArtifact.xpath('classifier/text()')
                artifact_classifier = classifier[0] if classifier else ''
                extension = snapshotArtifact.xpath('extension/text()')
                artifact_extension = extension[0] if extension else ''
                if artifact_classifier == artifact.classifier and artifact_extension == artifact.extension:
                    return self._uri_for_artifact(artifact, snapshotArtifact.xpath('value/text()')[0])
            timestamp_xmlpath = xml.xpath('/metadata/versioning/snapshot/timestamp/text()')
            if timestamp_xmlpath:
                timestamp = timestamp_xmlpath[0]
                build_number = xml.xpath('/metadata/versioning/snapshot/buildNumber/text()')[0]
                return self._uri_for_artifact(artifact, artifact.version.replace('SNAPSHOT', timestamp + '-' + build_number))
        return self._uri_for_artifact(artifact, artifact.version)

    def _uri_for_artifact(self, artifact, version=None):
        if artifact.is_snapshot() and (not version):
            raise ValueError('Expected uniqueversion for snapshot artifact ' + str(artifact))
        elif not artifact.is_snapshot():
            version = artifact.version
        if artifact.classifier:
            return posixpath.join(self.base, artifact.path(), artifact.artifact_id + '-' + version + '-' + artifact.classifier + '.' + artifact.extension)
        return posixpath.join(self.base, artifact.path(), artifact.artifact_id + '-' + version + '.' + artifact.extension)

    def _getContent(self, url, failmsg, force=True):
        if self.local:
            parsed_url = urlparse(url)
            if os.path.isfile(parsed_url.path):
                with io.open(parsed_url.path, 'rb') as f:
                    return f.read()
            if force:
                raise ValueError(failmsg + ' because can not find file: ' + url)
            return None
        response = self._request(url, failmsg, force)
        if response:
            return response.read()
        return None

    def _request(self, url, failmsg, force=True):
        url_to_use = url
        parsed_url = urlparse(url)
        if parsed_url.scheme == 's3':
            parsed_url = urlparse(url)
            bucket_name = parsed_url.netloc
            key_name = parsed_url.path[1:]
            client = boto3.client('s3', aws_access_key_id=self.module.params.get('username', ''), aws_secret_access_key=self.module.params.get('password', ''))
            url_to_use = client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': key_name}, ExpiresIn=10)
        req_timeout = self.module.params.get('timeout')
        self.module.params['url_username'] = self.module.params.get('username', '')
        self.module.params['url_password'] = self.module.params.get('password', '')
        self.module.params['http_agent'] = self.user_agent
        kwargs = {}
        if self.module.params['unredirected_headers']:
            kwargs['unredirected_headers'] = self.module.params['unredirected_headers']
        response, info = fetch_url(self.module, url_to_use, timeout=req_timeout, headers=self.headers, **kwargs)
        if info['status'] == 200:
            return response
        if force:
            raise ValueError(failmsg + ' because of ' + info['msg'] + 'for URL ' + url_to_use)
        return None

    def download(self, tmpdir, artifact, verify_download, filename=None, checksum_alg='md5'):
        if not artifact.version and (not artifact.version_by_spec) or artifact.version == 'latest':
            artifact = Artifact(artifact.group_id, artifact.artifact_id, self.find_latest_version_available(artifact), None, artifact.classifier, artifact.extension)
        url = self.find_uri_for_artifact(artifact)
        tempfd, tempname = tempfile.mkstemp(dir=tmpdir)
        try:
            if self.local:
                parsed_url = urlparse(url)
                if os.path.isfile(parsed_url.path):
                    shutil.copy2(parsed_url.path, tempname)
                else:
                    return 'Can not find local file: ' + parsed_url.path
            else:
                response = self._request(url, 'Failed to download artifact ' + str(artifact))
                with os.fdopen(tempfd, 'wb') as f:
                    shutil.copyfileobj(response, f)
            if verify_download:
                invalid_checksum = self.is_invalid_checksum(tempname, url, checksum_alg)
                if invalid_checksum:
                    os.remove(tempname)
                    return invalid_checksum
        except Exception as e:
            os.remove(tempname)
            raise e
        shutil.move(tempname, artifact.get_filename(filename))
        return None

    def is_invalid_checksum(self, file, remote_url, checksum_alg='md5'):
        if os.path.exists(file):
            local_checksum = self._local_checksum(checksum_alg, file)
            if self.local:
                parsed_url = urlparse(remote_url)
                remote_checksum = self._local_checksum(checksum_alg, parsed_url.path)
            else:
                try:
                    remote_checksum = to_text(self._getContent(remote_url + '.' + checksum_alg, 'Failed to retrieve checksum', False), errors='strict')
                except UnicodeError as e:
                    return 'Cannot retrieve a valid %s checksum from %s: %s' % (checksum_alg, remote_url, to_native(e))
                if not remote_checksum:
                    return 'Cannot find %s checksum from %s' % (checksum_alg, remote_url)
            try:
                _remote_checksum = remote_checksum.split(None, 1)[0]
                remote_checksum = _remote_checksum
            except IndexError:
                pass
            if local_checksum.lower() == remote_checksum.lower():
                return None
            else:
                return 'Checksum does not match: we computed ' + local_checksum + ' but the repository states ' + remote_checksum
        return 'Path does not exist: ' + file

    def _local_checksum(self, checksum_alg, file):
        if checksum_alg.lower() == 'md5':
            hash = hashlib.md5()
        elif checksum_alg.lower() == 'sha1':
            hash = hashlib.sha1()
        else:
            raise ValueError('Unknown checksum_alg %s' % checksum_alg)
        with io.open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash.update(chunk)
        return hash.hexdigest()