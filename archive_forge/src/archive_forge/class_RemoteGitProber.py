import os
import sys
from .. import __version__ as breezy_version  # noqa: F401
from .. import errors as brz_errors
from .. import trace, urlutils, version_info
from ..commands import plugin_cmds
from ..controldir import ControlDirFormat, Prober, format_registry
from ..controldir import \
from ..transport import (register_lazy_transport, register_transport_proto,
from ..revisionspec import RevisionSpec_dwim, revspec_registry
from ..hooks import install_lazy_named_hook
from ..location import hooks as location_hooks
from ..repository import format_registry as repository_format_registry
from ..repository import \
from ..branch import network_format_registry as branch_network_format_registry
from ..branch import format_registry as branch_format_registry
from ..workingtree import format_registry as workingtree_format_registry
from ..diff import format_registry as diff_format_registry
from ..send import format_registry as send_format_registry
from ..directory_service import directories
from ..help_topics import topic_registry
from ..foreign import foreign_vcs_registry
from ..config import Option, bool_from_store, option_registry
class RemoteGitProber(Prober):

    @classmethod
    def priority(klass, transport):
        if 'git' in transport.base:
            return -15
        return -10

    def probe_http_transport(self, transport):
        base_url = urlutils.strip_segment_parameters(transport.external_url())
        url = urlutils.URL.from_string(base_url)
        url.user = url.quoted_user = None
        url.password = url.quoted_password = None
        host = url.host
        url = urlutils.join(str(url), 'info/refs') + '?service=git-upload-pack'
        headers = {'Content-Type': 'application/x-git-upload-pack-request', 'Accept': 'application/x-git-upload-pack-result'}
        if is_github_url(url):
            headers['User-Agent'] = user_agent_for_github()
        resp = transport.request('GET', url, headers=headers)
        if resp.status in (404, 405):
            raise brz_errors.NotBranchError(transport.base)
        elif resp.status == 400 and resp.reason == 'no such method: info':
            raise brz_errors.NotBranchError(transport.base)
        elif resp.status != 200:
            raise brz_errors.UnexpectedHttpStatus(url, resp.status, headers=resp.getheaders())
        ct = resp.getheader('Content-Type')
        if ct and ct.startswith('application/x-git'):
            from .remote import RemoteGitControlDirFormat
            return RemoteGitControlDirFormat()
        elif not ct:
            from .dir import BareLocalGitControlDirFormat
            ret = BareLocalGitControlDirFormat()
            ret._refs_text = resp.read()
            return ret
        raise brz_errors.NotBranchError(transport.base)

    def probe_transport(self, transport):
        try:
            external_url = transport.external_url()
        except brz_errors.InProcessTransport:
            raise brz_errors.NotBranchError(path=transport.base)
        if external_url.startswith('http:') or external_url.startswith('https:'):
            return self.probe_http_transport(transport)
        if not external_url.startswith('git://') and (not external_url.startswith('git+')):
            raise brz_errors.NotBranchError(transport.base)
        from .remote import GitSmartTransport, RemoteGitControlDirFormat
        if isinstance(transport, GitSmartTransport):
            return RemoteGitControlDirFormat()
        raise brz_errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        from .remote import RemoteGitControlDirFormat
        return [RemoteGitControlDirFormat()]