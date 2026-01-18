import base64
import binascii
import logging
import bcrypt
import webob
from oslo_config import cfg
from oslo_middleware import base
def validate_auth_file(auth_file):
    """Read the auth user file and validate its correctness

    :param: auth_file: Path to user auth file
    :raises: ConfigInvalid on validation error
    """
    try:
        with open(auth_file, 'r') as f:
            for line in f:
                entry = line.strip()
                if entry and ':' in entry:
                    parse_entry(entry)
    except OSError:
        raise ConfigInvalid(error_msg='Problem reading auth user file')