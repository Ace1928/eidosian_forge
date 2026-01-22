from ansible.playbook.attribute import FieldAttribute
def _post_validate_delegate_to(self, attr, value, templar):
    """This method exists just to make it clear that ``Task.post_validate``
        does not template this value, it is set via ``TaskExecutor._calculate_delegate_to``
        """
    return value