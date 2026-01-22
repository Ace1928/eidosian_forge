import base64
import logging
import struct
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, unify_credentials
from spnego._gss import GSSAPIProxy
from spnego._spnego import (
from spnego._sspi import SSPIProxy
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
def _rebuild_context_list(self, mech_types: typing.Optional[typing.List[str]]=None, channel_bindings: typing.Optional[GssChannelBindings]=None) -> typing.List[str]:
    """Builds a new context list that are available to the client."""
    available_contexts = self._available_contexts or {}
    last_err = None
    if not available_contexts:
        context_kwargs: typing.Dict[str, typing.Any] = {'hostname': self._hostname, 'service': self._service, 'channel_bindings': self.channel_bindings, 'context_req': self.context_req}
        all_protocols = self._preferred_mech_list()
        for mech in all_protocols:
            if mech_types and mech.value not in mech_types:
                continue
            protocol = mech.name
            try:
                log.debug(f'Attempting to create {protocol} context when building SPNEGO mech list')
                options = self.options & ~NegotiateOptions.use_negotiate
                if protocol == 'ntlm' and 'ntlm' in SSPIProxy.available_protocols(options=options):
                    options |= NegotiateOptions.use_ntlm
                if self.usage == 'accept':
                    context = spnego.server(protocol=protocol, options=options, **context_kwargs)
                else:
                    context = spnego.client(self._credentials, protocol=protocol, options=options, **context_kwargs)
                context._is_wrapped = True
                available_contexts[mech] = context
            except Exception as e:
                last_err = e
                log.debug('Failed to create context for SPNEGO protocol %s: %s', protocol, str(e))
                continue
    self._context_list = {}
    mech_list = []
    for mech, context in available_contexts.items():
        try:
            first_token = context.step(channel_bindings=channel_bindings) if self.usage == 'initiate' else None
        except Exception as e:
            last_err = e
            log.debug('Failed to create first token for SPNEGO protocol %s: %s', mech.name, str(e))
            continue
        self._context_list[mech] = (context, first_token)
        mech_list.append(mech.value)
    if not mech_list:
        raise BadMechanismError(context_msg='Unable to negotiate common mechanism', base_error=last_err)
    return mech_list