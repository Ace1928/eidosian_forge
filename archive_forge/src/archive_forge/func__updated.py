import re
def _updated(self):
    """
        Assign to new_dict.updated to track updates
        """
    updated = self.updated
    if updated is not None:
        args = self.updated_args
        if args is None:
            args = (self,)
        updated(*args)