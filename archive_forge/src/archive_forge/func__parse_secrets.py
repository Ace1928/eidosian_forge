from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1.containers import CertificateContainer
from barbicanclient.v1.containers import Container
from barbicanclient.v1.containers import RSAContainer
@staticmethod
def _parse_secrets(secrets):
    if not secrets:
        raise ValueError('Must supply at least one secret.')
    return dict(((s.split('=')[0], s.split('=')[1]) for s in secrets if s.count('=') == 1))