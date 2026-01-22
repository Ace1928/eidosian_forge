import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
from ctypes.wintypes import (
from typing import TYPE_CHECKING, Any
from ._ssl_constants import _set_ssl_context_verify_mode
def _verify_using_custom_ca_certs(ssl_context: ssl.SSLContext, custom_ca_certs: list[bytes], hIntermediateCertStore: HCERTSTORE, pPeerCertContext: c_void_p, pChainPara: PCERT_CHAIN_PARA, server_hostname: str | None, chain_flags: int) -> None:
    hChainEngine = None
    hRootCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
    try:
        for cert_bytes in custom_ca_certs:
            CertAddEncodedCertificateToStore(hRootCertStore, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, cert_bytes, len(cert_bytes), CERT_STORE_ADD_USE_EXISTING, None)
        cert_chain_engine_config = CERT_CHAIN_ENGINE_CONFIG()
        cert_chain_engine_config.cbSize = sizeof(cert_chain_engine_config)
        cert_chain_engine_config.hExclusiveRoot = hRootCertStore
        pConfig = pointer(cert_chain_engine_config)
        phChainEngine = pointer(HCERTCHAINENGINE())
        CertCreateCertificateChainEngine(pConfig, phChainEngine)
        hChainEngine = phChainEngine.contents
        _get_and_verify_cert_chain(ssl_context, hChainEngine, hIntermediateCertStore, pPeerCertContext, pChainPara, server_hostname, chain_flags)
    finally:
        if hChainEngine:
            CertFreeCertificateChainEngine(hChainEngine)
        CertCloseStore(hRootCertStore, 0)