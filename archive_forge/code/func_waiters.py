import logging
from botocore import xform_name
@property
def waiters(self):
    """
        Get a list of waiters for this resource.

        :type: list(:py:class:`Waiter`)
        """
    waiters = []
    for name, item in self._definition.get('waiters', {}).items():
        name = self._get_name('waiter', Waiter.PREFIX + name)
        waiters.append(Waiter(name, item))
    return waiters