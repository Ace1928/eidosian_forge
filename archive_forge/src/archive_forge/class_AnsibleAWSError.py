from ansible.module_utils._text import to_native
class AnsibleAWSError(Exception):

    def __str__(self):
        if self.exception and self.message:
            return f'{self.message}: {to_native(self.exception)}'
        return super().__str__()

    def __init__(self, message=None, exception=None, **kwargs):
        if not message and (not exception):
            super().__init__()
        elif not message:
            super().__init__(exception)
        else:
            super().__init__(message)
        self.exception = exception
        self.message = message
        self.kwargs = kwargs or {}