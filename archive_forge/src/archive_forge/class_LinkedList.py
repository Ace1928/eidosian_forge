import weakref
from weakref import ReferenceType
class LinkedList(Generic[T]):
    """Specialized linked list implementation to support the deb822 parser needs

    We deliberately trade "encapsulation" for features needed by this library
    to facilitate their implementation.  Notably, we allow nodes to leak and assume
    well-behaved calls to remove_node - because that makes it easier to implement
    components like Deb822InvalidParagraphElement.
    """
    __slots__ = ('head_node', 'tail_node', '_size')

    def __init__(self, values=None):
        self.head_node = None
        self.tail_node = None
        self._size = 0
        if values is not None:
            self.extend(values)

    def __bool__(self):
        return self.head_node is not None

    def __len__(self):
        return self._size

    @property
    def tail(self):
        return self.tail_node.value if self.tail_node is not None else None

    def pop(self):
        if self.tail_node is None:
            raise IndexError('pop from empty list')
        self.remove_node(self.tail_node)

    def iter_nodes(self):
        head_node = self.head_node
        if head_node is None:
            return
        yield from head_node.iter_next()

    def __iter__(self):
        yield from (node.value for node in self.iter_nodes())

    def __reversed__(self):
        tail_node = self.tail_node
        if tail_node is None:
            return
        yield from (n.value for n in tail_node.iter_previous())

    def remove_node(self, node):
        if node is self.head_node:
            self.head_node = node.next_node
            if self.head_node is None:
                self.tail_node = None
        elif node is self.tail_node:
            self.tail_node = node.previous_node
            assert self.tail_node is not None
        assert self._size > 0
        self._size -= 1
        node.remove()

    def insert_at_head(self, value):
        if self.head_node is None:
            return self.append(value)
        return self.insert_before(value, self.head_node)

    def append(self, value):
        node = LinkedListNode(value)
        if self.head_node is None:
            self.head_node = node
            self.tail_node = node
        else:
            assert self.tail_node is not None
            assert self.tail_node is not node
            node.previous_node = self.tail_node
            self.tail_node.next_node = node
            self.tail_node = node
        self._size += 1
        return node

    def insert_before(self, value, existing_node):
        return self.insert_node_before(LinkedListNode(value), existing_node)

    def insert_after(self, value, existing_node):
        return self.insert_node_after(LinkedListNode(value), existing_node)

    def insert_node_before(self, new_node, existing_node):
        if self.head_node is None:
            raise ValueError('List is empty; node argument cannot be valid')
        if new_node.next_node is not None or new_node.previous_node is not None:
            raise ValueError('New node must not already be inserted!')
        existing_node.insert_before(new_node)
        if existing_node is self.head_node:
            self.head_node = new_node
        self._size += 1
        return new_node

    def insert_node_after(self, new_node, existing_node):
        if self.tail_node is None:
            raise ValueError('List is empty; node argument cannot be valid')
        if new_node.next_node is not None or new_node.previous_node is not None:
            raise ValueError('New node must not already be inserted!')
        existing_node.insert_after(new_node)
        if existing_node is self.tail_node:
            self.tail_node = new_node
        self._size += 1
        return new_node

    def extend(self, values):
        for v in values:
            self.append(v)

    def clear(self):
        self.head_node = None
        self.tail_node = None
        self._size = 0