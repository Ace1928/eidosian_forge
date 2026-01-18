from pyomo.core.kernel.base import (

        Generates an efficient traversal of all components
        stored under this container. Components are
        categorized objects that are either (1) not
        containers, or (2) are heterogeneous containers.

        Args:
            ctype: Indicates the category of components to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.
            descend_into (bool, function): Indicates whether
                or not to descend into a heterogeneous
                container. Default is True, which is
                equivalent to `lambda x: True`, meaning all
                heterogeneous containers will be descended
                into.

        Returns:
            iterator of components in the storage tree
        