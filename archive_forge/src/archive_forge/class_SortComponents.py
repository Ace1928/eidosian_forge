import enum
import sys
class SortComponents(enum.Flag, **strictEnum):
    """
    This class is a convenient wrapper for specifying various sort
    ordering.  We pass these objects to the "sort" argument to various
    accessors / iterators to control how much work we perform sorting
    the resultant list.  The idea is that
    "sort=SortComponents.deterministic" is more descriptive than
    "sort=True".
    """
    UNSORTED = 0
    ORDERED_INDICES = 2
    SORTED_INDICES = 4
    ALPHABETICAL = 8
    unsorted = UNSORTED
    indices = SORTED_INDICES
    declOrder = UNSORTED
    declarationOrder = declOrder
    alphaOrder = ALPHABETICAL
    alphabeticalOrder = alphaOrder
    alphabetical = alphaOrder
    deterministic = ORDERED_INDICES
    sortBoth = indices | alphabeticalOrder
    alphabetizeComponentAndIndex = sortBoth

    @classmethod
    def _missing_(cls, value):
        if type(value) is bool:
            if value:
                return cls.SORTED_INDICES | cls.ALPHABETICAL
            else:
                return cls.UNSORTED
        elif value is None:
            return cls.UNSORTED
        return super()._missing_(value)

    @staticmethod
    def default():
        return SortComponents.UNSORTED

    @staticmethod
    def sorter(sort_by_names=False, sort_by_keys=False):
        sort = SortComponents.default()
        if sort_by_names:
            sort |= SortComponents.ALPHABETICAL
        if sort_by_keys:
            sort |= SortComponents.SORTED_INDICES
        return sort

    @staticmethod
    def sort_names(flag):
        return SortComponents.ALPHABETICAL in SortComponents(flag)

    @staticmethod
    def sort_indices(flag):
        return SortComponents.SORTED_INDICES in SortComponents(flag)