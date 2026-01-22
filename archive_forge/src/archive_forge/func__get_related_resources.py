import logging
from botocore import xform_name
def _get_related_resources(self, subresources):
    """
        Get a list of sub-resources or references.

        :type subresources: bool
        :param subresources: ``True`` to get sub-resources, ``False`` to
                             get references.
        :rtype: list(:py:class:`ResponseResource`)
        """
    resources = []
    for name, definition in self._get_has_definition().items():
        if subresources:
            name = self._get_name('subresource', name, snake_case=False)
        else:
            name = self._get_name('reference', name)
        action = Action(name, definition, self._resource_defs)
        data_required = False
        for identifier in action.resource.identifiers:
            if identifier.source == 'data':
                data_required = True
                break
        if subresources and (not data_required):
            resources.append(action)
        elif not subresources and data_required:
            resources.append(action)
    return resources