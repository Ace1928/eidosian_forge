from pyomo.core.kernel.base import ICategorizedObjectContainer

        Generates an efficient traversal of all components
        stored under this container. Components are
        categorized objects that are either (1) not
        containers, or (2) are heterogeneous containers.

        Args:
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.

        Returns:
            iterator of components in the storage tree
        