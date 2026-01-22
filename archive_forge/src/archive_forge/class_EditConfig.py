from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations import util
from .errors import OperationError
import logging
class EditConfig(RPC):
    """`edit-config` RPC"""

    def request(self, config, format='xml', target='candidate', default_operation=None, test_option=None, error_option=None):
        """Loads all or part of the specified *config* to the *target* configuration datastore.

        *target* is the name of the configuration datastore being edited

        *config* is the configuration, which must be rooted in the `config` element. It can be specified either as a string or an :class:`~xml.etree.ElementTree.Element`.

        *default_operation* if specified must be one of { `"merge"`, `"replace"`, or `"none"` }

        *test_option* if specified must be one of { `"test-then-set"`, `"set"`, `"test-only"` }

        *error_option* if specified must be one of { `"stop-on-error"`, `"continue-on-error"`, `"rollback-on-error"` }

        The `"rollback-on-error"` *error_option* depends on the `:rollback-on-error` capability.
        """
        node = new_ele('edit-config')
        node.append(util.datastore_or_url('target', target, self._assert))
        if default_operation is not None and util.validate_args('default_operation', default_operation, ['merge', 'replace', 'none']) is True:
            sub_ele(node, 'default-operation').text = default_operation
        if test_option is not None and util.validate_args('test_option', test_option, ['test-then-set', 'set', 'test-only']) is True:
            self._assert(':validate')
            if test_option == 'test-only':
                self._assert(':validate:1.1')
            sub_ele(node, 'test-option').text = test_option
        if error_option is not None and util.validate_args('error_option', error_option, ['stop-on-error', 'continue-on-error', 'rollback-on-error']) is True:
            if error_option == 'rollback-on-error':
                self._assert(':rollback-on-error')
            sub_ele(node, 'error-option').text = error_option
        if format == 'xml':
            node.append(validated_element(config, ('config', qualify('config'))))
        elif format == 'text':
            config_text = sub_ele(node, 'config-text')
            sub_ele(config_text, 'configuration-text').text = config
        elif format == 'url':
            if util.url_validator(config):
                self._assert(':url')
                sub_ele(node, 'url').text = config
            else:
                raise OperationError('Invalid URL.')
        node = self._device_handler.transform_edit_config(node)
        return self._request(node)