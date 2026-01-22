from ..utils.orderedtype import OrderedType
def InputField(self):
    """
        Mount the UnmountedType as InputField
        """
    from .inputfield import InputField
    return self.mount_as(InputField)