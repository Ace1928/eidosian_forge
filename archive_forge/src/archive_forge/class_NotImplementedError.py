from octavia_lib.i18n import _
class NotImplementedError(Exception):
    """Exception raised when a driver does not implement an API function.

    :param user_fault_string: String provided to the API requester.
    :type user_fault_string: string
    :param operator_fault_string: Optional string logged by the Octavia API
      for the operator to use when debugging.
    :type operator_fault_string: string
    """
    user_fault_string = _('This feature is not implemented by the provider.')
    operator_fault_string = _('This feature is not implemented by this provider.')

    def __init__(self, *args, **kwargs):
        self.user_fault_string = kwargs.pop('user_fault_string', self.user_fault_string)
        self.operator_fault_string = kwargs.pop('operator_fault_string', self.operator_fault_string)
        super().__init__(self.user_fault_string, *args, **kwargs)