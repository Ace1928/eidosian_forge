from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
class PythonPVector(object):
    """
    Support structure for PVector that implements structural sharing for vectors using a trie.
    """
    __slots__ = ('_count', '_shift', '_root', '_tail', '_tail_offset', '__weakref__')

    def __new__(cls, count, shift, root, tail):
        self = super(PythonPVector, cls).__new__(cls)
        self._count = count
        self._shift = shift
        self._root = root
        self._tail = tail
        self._tail_offset = self._count - len(self._tail)
        return self

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.start is None and index.stop is None and (index.step is None):
                return self
            return _EMPTY_PVECTOR.extend(self.tolist()[index])
        if index < 0:
            index += self._count
        return PythonPVector._node_for(self, index)[index & BIT_MASK]

    def __add__(self, other):
        return self.extend(other)

    def __repr__(self):
        return 'pvector({0})'.format(str(self.tolist()))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return iter(self.tolist())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return self is other or ((hasattr(other, '__len__') and self._count == len(other)) and compare_pvector(self, other, operator.eq))

    def __gt__(self, other):
        return compare_pvector(self, other, operator.gt)

    def __lt__(self, other):
        return compare_pvector(self, other, operator.lt)

    def __ge__(self, other):
        return compare_pvector(self, other, operator.ge)

    def __le__(self, other):
        return compare_pvector(self, other, operator.le)

    def __mul__(self, times):
        if times <= 0 or self is _EMPTY_PVECTOR:
            return _EMPTY_PVECTOR
        if times == 1:
            return self
        return _EMPTY_PVECTOR.extend(times * self.tolist())
    __rmul__ = __mul__

    def _fill_list(self, node, shift, the_list):
        if shift:
            shift -= SHIFT
            for n in node:
                self._fill_list(n, shift, the_list)
        else:
            the_list.extend(node)

    def tolist(self):
        """
        The fastest way to convert the vector into a python list.
        """
        the_list = []
        self._fill_list(self._root, self._shift, the_list)
        the_list.extend(self._tail)
        return the_list

    def _totuple(self):
        """
        Returns the content as a python tuple.
        """
        return tuple(self.tolist())

    def __hash__(self):
        return hash(self._totuple())

    def transform(self, *transformations):
        return transform(self, transformations)

    def __reduce__(self):
        return (pvector, (self.tolist(),))

    def mset(self, *args):
        if len(args) % 2:
            raise TypeError('mset expected an even number of arguments')
        evolver = self.evolver()
        for i in range(0, len(args), 2):
            evolver[args[i]] = args[i + 1]
        return evolver.persistent()

    class Evolver(object):
        __slots__ = ('_count', '_shift', '_root', '_tail', '_tail_offset', '_dirty_nodes', '_extra_tail', '_cached_leafs', '_orig_pvector')

        def __init__(self, v):
            self._reset(v)

        def __getitem__(self, index):
            if not isinstance(index, Integral):
                raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)
            if index < 0:
                index += self._count + len(self._extra_tail)
            if self._count <= index < self._count + len(self._extra_tail):
                return self._extra_tail[index - self._count]
            return PythonPVector._node_for(self, index)[index & BIT_MASK]

        def _reset(self, v):
            self._count = v._count
            self._shift = v._shift
            self._root = v._root
            self._tail = v._tail
            self._tail_offset = v._tail_offset
            self._dirty_nodes = {}
            self._cached_leafs = {}
            self._extra_tail = []
            self._orig_pvector = v

        def append(self, element):
            self._extra_tail.append(element)
            return self

        def extend(self, iterable):
            self._extra_tail.extend(iterable)
            return self

        def set(self, index, val):
            self[index] = val
            return self

        def __setitem__(self, index, val):
            if not isinstance(index, Integral):
                raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)
            if index < 0:
                index += self._count + len(self._extra_tail)
            if 0 <= index < self._count:
                node = self._cached_leafs.get(index >> SHIFT)
                if node:
                    node[index & BIT_MASK] = val
                elif index >= self._tail_offset:
                    if id(self._tail) not in self._dirty_nodes:
                        self._tail = list(self._tail)
                        self._dirty_nodes[id(self._tail)] = True
                        self._cached_leafs[index >> SHIFT] = self._tail
                    self._tail[index & BIT_MASK] = val
                else:
                    self._root = self._do_set(self._shift, self._root, index, val)
            elif self._count <= index < self._count + len(self._extra_tail):
                self._extra_tail[index - self._count] = val
            elif index == self._count + len(self._extra_tail):
                self._extra_tail.append(val)
            else:
                raise IndexError('Index out of range: %s' % (index,))

        def _do_set(self, level, node, i, val):
            if id(node) in self._dirty_nodes:
                ret = node
            else:
                ret = list(node)
                self._dirty_nodes[id(ret)] = True
            if level == 0:
                ret[i & BIT_MASK] = val
                self._cached_leafs[i >> SHIFT] = ret
            else:
                sub_index = i >> level & BIT_MASK
                ret[sub_index] = self._do_set(level - SHIFT, node[sub_index], i, val)
            return ret

        def delete(self, index):
            del self[index]
            return self

        def __delitem__(self, key):
            if self._orig_pvector:
                l = PythonPVector(self._count, self._shift, self._root, self._tail).tolist()
                l.extend(self._extra_tail)
                self._reset(_EMPTY_PVECTOR)
                self._extra_tail = l
            del self._extra_tail[key]

        def persistent(self):
            result = self._orig_pvector
            if self.is_dirty():
                result = PythonPVector(self._count, self._shift, self._root, self._tail).extend(self._extra_tail)
                self._reset(result)
            return result

        def __len__(self):
            return self._count + len(self._extra_tail)

        def is_dirty(self):
            return bool(self._dirty_nodes or self._extra_tail)

    def evolver(self):
        return PythonPVector.Evolver(self)

    def set(self, i, val):
        if not isinstance(i, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(i).__name__)
        if i < 0:
            i += self._count
        if 0 <= i < self._count:
            if i >= self._tail_offset:
                new_tail = list(self._tail)
                new_tail[i & BIT_MASK] = val
                return PythonPVector(self._count, self._shift, self._root, new_tail)
            return PythonPVector(self._count, self._shift, self._do_set(self._shift, self._root, i, val), self._tail)
        if i == self._count:
            return self.append(val)
        raise IndexError('Index out of range: %s' % (i,))

    def _do_set(self, level, node, i, val):
        ret = list(node)
        if level == 0:
            ret[i & BIT_MASK] = val
        else:
            sub_index = i >> level & BIT_MASK
            ret[sub_index] = self._do_set(level - SHIFT, node[sub_index], i, val)
        return ret

    @staticmethod
    def _node_for(pvector_like, i):
        if 0 <= i < pvector_like._count:
            if i >= pvector_like._tail_offset:
                return pvector_like._tail
            node = pvector_like._root
            for level in range(pvector_like._shift, 0, -SHIFT):
                node = node[i >> level & BIT_MASK]
            return node
        raise IndexError('Index out of range: %s' % (i,))

    def _create_new_root(self):
        new_shift = self._shift
        if self._count >> SHIFT > 1 << self._shift:
            new_root = [self._root, self._new_path(self._shift, self._tail)]
            new_shift += SHIFT
        else:
            new_root = self._push_tail(self._shift, self._root, self._tail)
        return (new_root, new_shift)

    def append(self, val):
        if len(self._tail) < BRANCH_FACTOR:
            new_tail = list(self._tail)
            new_tail.append(val)
            return PythonPVector(self._count + 1, self._shift, self._root, new_tail)
        new_root, new_shift = self._create_new_root()
        return PythonPVector(self._count + 1, new_shift, new_root, [val])

    def _new_path(self, level, node):
        if level == 0:
            return node
        return [self._new_path(level - SHIFT, node)]

    def _mutating_insert_tail(self):
        self._root, self._shift = self._create_new_root()
        self._tail = []

    def _mutating_fill_tail(self, offset, sequence):
        max_delta_len = BRANCH_FACTOR - len(self._tail)
        delta = sequence[offset:offset + max_delta_len]
        self._tail.extend(delta)
        delta_len = len(delta)
        self._count += delta_len
        return offset + delta_len

    def _mutating_extend(self, sequence):
        offset = 0
        sequence_len = len(sequence)
        while offset < sequence_len:
            offset = self._mutating_fill_tail(offset, sequence)
            if len(self._tail) == BRANCH_FACTOR:
                self._mutating_insert_tail()
        self._tail_offset = self._count - len(self._tail)

    def extend(self, obj):
        l = obj.tolist() if isinstance(obj, PythonPVector) else list(obj)
        if l:
            new_vector = self.append(l[0])
            new_vector._mutating_extend(l[1:])
            return new_vector
        return self

    def _push_tail(self, level, parent, tail_node):
        """
        if parent is leaf, insert node,
        else does it map to an existing child? ->
             node_to_insert = push node one more level
        else alloc new path

        return  node_to_insert placed in copy of parent
        """
        ret = list(parent)
        if level == SHIFT:
            ret.append(tail_node)
            return ret
        sub_index = self._count - 1 >> level & BIT_MASK
        if len(parent) > sub_index:
            ret[sub_index] = self._push_tail(level - SHIFT, parent[sub_index], tail_node)
            return ret
        ret.append(self._new_path(level - SHIFT, tail_node))
        return ret

    def index(self, value, *args, **kwargs):
        return self.tolist().index(value, *args, **kwargs)

    def count(self, value):
        return self.tolist().count(value)

    def delete(self, index, stop=None):
        l = self.tolist()
        del l[_index_or_slice(index, stop)]
        return _EMPTY_PVECTOR.extend(l)

    def remove(self, value):
        l = self.tolist()
        l.remove(value)
        return _EMPTY_PVECTOR.extend(l)