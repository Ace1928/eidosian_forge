from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
class ModelVisitor:
    """
    A visitor design pattern class that can be used for validating XML data related to an XSD
    model group. The visit of the model is done using an external match information,
    counting the occurrences and yielding tuples in case of model's item occurrence errors.
    Ends setting the current element to `None`.

    :param root: the root model group.
    :ivar occurs: the Counter instance for keeping track of occurrences of XSD elements and groups.
    :ivar element: the current XSD element, initialized to the first element of the model.
    :ivar group: the current XSD model group, initialized to *root* argument.
    :ivar items: the current XSD group's items iterator.
    :ivar match: if the XSD group has an effective item match.
    """
    _groups: List[Tuple[ModelGroupType, Iterator[ModelParticleType], bool]]
    element: Optional[SchemaElementType]
    __slots__ = ('_groups', 'root', 'occurs', 'element', 'group', 'items', 'match')

    def __init__(self, root: ModelGroupType) -> None:
        self._groups = []
        self.root = root
        self.occurs = Counter[Union[ModelParticleType, Tuple[ModelParticleType]]]()
        self.element = None
        self.group = root
        self.items = self.iter_group()
        self.match = False
        self._start()

    def __repr__(self) -> str:
        return '%s(root=%r)' % (self.__class__.__name__, self.root)

    def clear(self) -> None:
        del self._groups[:]
        self.occurs.clear()
        self.element = None
        self.group = self.root
        self.items = self.iter_group()
        self.match = False

    def _start(self) -> None:
        while True:
            item = next(self.items, None)
            if item is None:
                if not self._groups:
                    break
                self.group, self.items, self.match = self._groups.pop()
            elif not isinstance(item, groups.XsdGroup):
                self.element = item
                break
            elif item:
                self._groups.append((self.group, self.items, self.match))
                self.group = item
                self.items = self.iter_group()
                self.match = False

    @property
    def expected(self) -> List[SchemaElementType]:
        """
        Returns the expected elements of the current and descendant groups.
        """
        expected: List[SchemaElementType] = []
        items: Union[ModelGroupType, Iterator[ModelParticleType]]
        if self.group.model == 'choice':
            items = self.group
        elif self.group.model == 'all':
            items = (e for e in self.group if e.min_occurs > self.occurs[e])
        else:
            items = (e for e in self.group if e.min_occurs > self.occurs[e])
        for e in items:
            if isinstance(e, groups.XsdGroup):
                expected.extend(e.iter_elements())
            else:
                expected.append(e)
                expected.extend(e.maps.substitution_groups.get(e.name or '', ()))
        return expected

    def restart(self) -> None:
        self.clear()
        self._start()

    def stop(self) -> Iterator[AdvanceYieldedType]:
        while self.element is not None:
            for e in self.advance():
                yield e

    def iter_group(self) -> Iterator[ModelParticleType]:
        """Returns an iterator for the current model group."""
        if self.group.max_occurs == 0:
            return iter(())
        elif self.group.model != 'all':
            return iter(self.group)
        else:
            return (e for e in self.group.iter_elements() if not e.is_over(self.occurs[e]))

    def advance(self, match: bool=False) -> Iterator[AdvanceYieldedType]:
        """
        Generator function for advance to the next element. Yields tuples with
        particles information when occurrence violation is found.

        :param match: provides current element match.
        """

        def stop_item(item: ModelParticleType) -> bool:
            """
            Stops element or group matching, incrementing current group counter.

            :return: `True` if the item has violated the minimum occurrences for itself             or for the current group, `False` otherwise.
            """
            if isinstance(item, groups.XsdGroup):
                self.group, self.items, self.match = self._groups.pop()
            if self.group.model == 'choice':
                item_occurs = occurs[item]
                if not item_occurs:
                    return False
                item_max_occurs = occurs[item,] or item_occurs
                if item.max_occurs is None:
                    min_group_occurs = 1
                elif item_occurs % item.max_occurs:
                    min_group_occurs = 1 + item_occurs // item.max_occurs
                else:
                    min_group_occurs = item_occurs // item.max_occurs
                max_group_occurs = max(1, item_max_occurs // (item.min_occurs or 1))
                occurs[self.group] += min_group_occurs
                occurs[self.group,] += max_group_occurs
                occurs[item] = 0
                self.items = self.iter_group()
                self.match = False
                return item.is_missing(item_max_occurs)
            elif self.group.model == 'all':
                return False
            elif self.match:
                pass
            elif occurs[item]:
                self.match = True
            elif item.is_emptiable():
                return False
            elif self._groups:
                return stop_item(self.group)
            elif self.group.min_occurs <= max(occurs[self.group], occurs[self.group,]):
                return stop_item(self.group)
            else:
                return True
            if item is self.group[-1]:
                for k, item2 in enumerate(self.group, start=1):
                    item_occurs = occurs[item2]
                    if not item_occurs:
                        continue
                    item_max_occurs = occurs[item2,] or item_occurs
                    if item_max_occurs == 1 or any((not x.is_emptiable() for x in self.group[k:])):
                        occurs[self.group] += 1
                        break
                    min_group_occurs = max(1, item_occurs // (item2.max_occurs or item_occurs))
                    max_group_occurs = max(1, item_max_occurs // (item2.min_occurs or 1))
                    occurs[self.group] += min_group_occurs
                    occurs[self.group,] += max_group_occurs
                    break
            return item.is_missing(max(occurs[item], occurs[item,]))
        occurs = self.occurs
        if self.element is None:
            raise XMLSchemaValueError('cannot advance, %r is ended!' % self)
        if match:
            occurs[self.element] += 1
            self.match = True
            if self.group.model == 'all':
                self.items = (e for e in self.group.iter_elements() if not e.is_over(occurs[e]))
            elif not self.element.is_over(occurs[self.element]):
                return
            elif self.group.model == 'choice' and self.element.is_ambiguous():
                return
        obj = None
        try:
            element_occurs = occurs[self.element]
            if stop_item(self.element):
                yield (self.element, element_occurs, [self.element])
            while True:
                while self.group.is_over(max(occurs[self.group], occurs[self.group,])):
                    stop_item(self.group)
                obj = next(self.items, None)
                if isinstance(obj, groups.XsdGroup):
                    self._groups.append((self.group, self.items, self.match))
                    self.group = obj
                    self.items = self.iter_group()
                    self.match = False
                    occurs[obj] = occurs[obj,] = 0
                elif obj is not None:
                    self.element = obj
                    if self.group.model == 'sequence':
                        occurs[obj] = 0
                    return
                elif not self.match:
                    if self.group.model == 'all':
                        if all((e.min_occurs <= occurs[e] for e in self.group.iter_elements())):
                            occurs[self.group] = 1
                    group, expected = (self.group, self.expected)
                    if stop_item(group) and expected:
                        yield (group, occurs[group], expected)
                elif self.group.model != 'all':
                    self.items, self.match = (self.iter_group(), False)
                elif any((e.min_occurs > occurs[e] for e in self.group.iter_elements())):
                    if not self.group.min_occurs:
                        yield (self.group, occurs[self.group], self.expected)
                    self.group, self.items, self.match = self._groups.pop()
                elif any((not e.is_over(occurs[e]) for e in self.group)):
                    self.items = self.iter_group()
                    self.match = False
                else:
                    occurs[self.group] = 1
        except IndexError:
            self.element = None
            if self.group.is_missing(max(occurs[self.group], occurs[self.group,])):
                if self.group.model == 'choice':
                    yield (self.group, occurs[self.group], self.expected)
                elif self.group.model == 'sequence':
                    if obj is not None:
                        yield (self.group, occurs[self.group], self.expected)
                elif any((e.min_occurs > occurs[e] for e in self.group)):
                    yield (self.group, occurs[self.group], self.expected)
            elif self.group.max_occurs is not None and self.group.max_occurs < occurs[self.group]:
                yield (self.group, occurs[self.group], self.expected)

    def iter_unordered_content(self, content: EncodedContentType, default_namespace: Optional[str]=None) -> Iterator[ContentItemType]:
        return iter_unordered_content(content, self.root, default_namespace)

    def iter_collapsed_content(self, content: Iterable[ContentItemType], default_namespace: Optional[str]=None) -> Iterator[ContentItemType]:
        return iter_collapsed_content(content, self.root, default_namespace)