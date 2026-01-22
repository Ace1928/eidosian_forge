from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class CredConfigGenerator(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for generating Credential Config files."""

    def __init__(self, config_type):
        self.config_type = config_type

    def get_token_type(self, subject_token_type):
        """Returns the type of token that this credential config uses."""
        default_token_type = 'urn:ietf:params:oauth:token-type:jwt'
        if self.config_type is ConfigType.WORKFORCE_POOLS:
            default_token_type = 'urn:ietf:params:oauth:token-type:id_token'
        return subject_token_type or default_token_type

    def _get_format(self, credential_source_type, credential_source_field_name):
        """Returns an optional dictionary indicating the format of the token.

    This is a shared method, that several different token types need access to.

    Args:
      credential_source_type: The format of the token, either 'json' or 'text'.
      credential_source_field_name: The field name of a JSON object containing
        the text version of the token.

    Raises:
      GeneratorError: if an invalid token format is specified, or no field name
      is specified for a json token.

    """
        if not credential_source_type:
            return None
        credential_source_type = credential_source_type.lower()
        if credential_source_type not in ('json', 'text'):
            raise GeneratorError('--credential-source-type must be either "json" or "text"')
        token_format = {'type': credential_source_type}
        if credential_source_type == 'json':
            if not credential_source_field_name:
                raise GeneratorError('--credential-source-field-name required for JSON formatted tokens')
            token_format['subject_token_field_name'] = credential_source_field_name
        return token_format

    def _format_already_defined(self, credential_source_type):
        if credential_source_type:
            raise GeneratorError('--credential-source-type is not supported with --azure or --aws')

    @abc.abstractmethod
    def get_source(self, args):
        """Gets the credential source info used for this credential config."""
        pass