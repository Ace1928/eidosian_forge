import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
class AcceptorConfig:

    def __init__(self, config):
        self.state = config['state']
        self.matcher = config['matcher']
        self.expected = config['expected']
        self.argument = config.get('argument')
        self.matcher_func = self._create_matcher_func()

    @property
    def explanation(self):
        if self.matcher == 'path':
            return 'For expression "{}" we matched expected path: "{}"'.format(self.argument, self.expected)
        elif self.matcher == 'pathAll':
            return 'For expression "%s" all members matched excepted path: "%s"' % (self.argument, self.expected)
        elif self.matcher == 'pathAny':
            return 'For expression "%s" we matched expected path: "%s" at least once' % (self.argument, self.expected)
        elif self.matcher == 'status':
            return 'Matched expected HTTP status code: %s' % self.expected
        elif self.matcher == 'error':
            return 'Matched expected service error code: %s' % self.expected
        else:
            return 'No explanation for unknown waiter type: "%s"' % self.matcher

    def _create_matcher_func(self):
        if self.matcher == 'path':
            return self._create_path_matcher()
        elif self.matcher == 'pathAll':
            return self._create_path_all_matcher()
        elif self.matcher == 'pathAny':
            return self._create_path_any_matcher()
        elif self.matcher == 'status':
            return self._create_status_matcher()
        elif self.matcher == 'error':
            return self._create_error_matcher()
        else:
            raise WaiterConfigError(error_msg='Unknown acceptor: %s' % self.matcher)

    def _create_path_matcher(self):
        expression = jmespath.compile(self.argument)
        expected = self.expected

        def acceptor_matches(response):
            if is_valid_waiter_error(response):
                return
            return expression.search(response) == expected
        return acceptor_matches

    def _create_path_all_matcher(self):
        expression = jmespath.compile(self.argument)
        expected = self.expected

        def acceptor_matches(response):
            if is_valid_waiter_error(response):
                return
            result = expression.search(response)
            if not isinstance(result, list) or not result:
                return False
            for element in result:
                if element != expected:
                    return False
            return True
        return acceptor_matches

    def _create_path_any_matcher(self):
        expression = jmespath.compile(self.argument)
        expected = self.expected

        def acceptor_matches(response):
            if is_valid_waiter_error(response):
                return
            result = expression.search(response)
            if not isinstance(result, list) or not result:
                return False
            for element in result:
                if element == expected:
                    return True
            return False
        return acceptor_matches

    def _create_status_matcher(self):
        expected = self.expected

        def acceptor_matches(response):
            status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            return status_code == expected
        return acceptor_matches

    def _create_error_matcher(self):
        expected = self.expected

        def acceptor_matches(response):
            return response.get('Error', {}).get('Code', '') == expected
        return acceptor_matches