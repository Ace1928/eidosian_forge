import typing
from spnego._context import ContextProxy, ContextReq
from spnego._credential import Credential, NTLMHash, unify_credentials
from spnego._credssp import CredSSPProxy
from spnego._gss import GSSAPIProxy
from spnego._negotiate import NegotiateProxy
from spnego._ntlm import NTLMProxy
from spnego._sspi import SSPIProxy
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import NegotiateOptions
def _new_context(username: typing.Optional[typing.Union[str, Credential, typing.List[Credential]]], password: typing.Optional[str], hostname: str, service: str, channel_bindings: typing.Optional[GssChannelBindings], context_req: ContextReq, protocol: str, options: NegotiateOptions, usage: str, **kwargs: typing.Any) -> ContextProxy:
    proto = protocol.lower()
    credentials = unify_credentials(username, password)
    sspi_protocols = SSPIProxy.available_protocols(options=options)
    gssapi_protocols = GSSAPIProxy.available_protocols(options=options)
    use_flags = NegotiateOptions.use_sspi | NegotiateOptions.use_gssapi | NegotiateOptions.use_negotiate | NegotiateOptions.use_ntlm
    use_specified = options & use_flags != 0
    sspi_remove = set()
    for cred in credentials:
        if isinstance(cred, NTLMHash):
            sspi_remove.add('negotiate')
            sspi_remove.add('ntlm')
    if sspi_remove:
        for protocol in sspi_remove:
            if protocol in sspi_protocols:
                sspi_protocols.remove(protocol)
    proxy: typing.Type[typing.Union[CredSSPProxy, NTLMProxy, SSPIProxy, GSSAPIProxy, NegotiateProxy]]
    if proto == 'credssp':
        proxy = CredSSPProxy
    elif options & NegotiateOptions.use_sspi or (not use_specified and proto in sspi_protocols):
        proxy = SSPIProxy
    elif options & NegotiateOptions.use_gssapi or (not use_specified and (proto == 'kerberos' or proto in gssapi_protocols)):
        proxy = GSSAPIProxy
    elif options & NegotiateOptions.use_negotiate or (not use_specified and proto == 'negotiate'):
        proxy = NegotiateProxy
    elif options & NegotiateOptions.use_ntlm or (not use_specified and proto == 'ntlm'):
        proto = 'ntlm' if proto == 'negotiate' else proto
        proxy = NTLMProxy
    else:
        raise ValueError("Invalid protocol specified '%s', must be kerberos, negotiate, or ntlm" % protocol)
    return proxy(username=credentials, password=None, hostname=hostname, service=service, channel_bindings=channel_bindings, context_req=context_req, usage=usage, protocol=proto, options=options, **kwargs)