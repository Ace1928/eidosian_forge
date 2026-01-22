import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from oslo_serialization import base64
from oslo_serialization import jsonutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
def args_array_to_patch(op, attributes):
    patch = []
    for attr in attributes:
        if not attr.startswith('/'):
            attr = '/' + attr
        if op in ['add', 'replace']:
            path, value = split_and_deserialize(attr)
            if path == '/labels' or path == '/health_status_reason':
                a = []
                a.append(value)
                value = str(handle_labels(a))
                patch.append({'op': op, 'path': path, 'value': value})
            else:
                patch.append({'op': op, 'path': path, 'value': value})
        elif op == 'remove':
            patch.append({'op': op, 'path': attr})
        else:
            raise exc.CommandError(_('Unknown PATCH operation: %s') % op)
    return patch