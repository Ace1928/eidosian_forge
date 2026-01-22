import logging
from botocore import xform_name
@property
def actions(self):
    """
        Get a list of actions for this resource.

        :type: list(:py:class:`Action`)
        """
    actions = []
    for name, item in self._definition.get('actions', {}).items():
        name = self._get_name('action', name)
        actions.append(Action(name, item, self._resource_defs))
    return actions