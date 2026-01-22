import docutils.utils.math.tex2unichar as tex2unichar
class math:
    """Base class for MathML elements."""
    nchildren = 1000000
    'Required number of children'

    def __init__(self, children=None, inline=None):
        """math([children]) -> MathML element

        children can be one child or a list of children."""
        self.children = []
        if children is not None:
            if type(children) is list:
                for child in children:
                    self.append(child)
            else:
                self.append(children)
        if inline is not None:
            self.inline = inline

    def __repr__(self):
        if hasattr(self, 'children'):
            return self.__class__.__name__ + '(%s)' % ','.join([repr(child) for child in self.children])
        else:
            return self.__class__.__name__

    def full(self):
        """Room for more children?"""
        return len(self.children) >= self.nchildren

    def append(self, child):
        """append(child) -> element

        Appends child and returns self if self is not full or first
        non-full parent."""
        assert not self.full()
        self.children.append(child)
        child.parent = self
        node = self
        while node.full():
            node = node.parent
        return node

    def delete_child(self):
        """delete_child() -> child

        Delete last child and return it."""
        child = self.children[-1]
        del self.children[-1]
        return child

    def close(self):
        """close() -> parent

        Close element and return first non-full element."""
        parent = self.parent
        while parent.full():
            parent = parent.parent
        return parent

    def xml(self):
        """xml() -> xml-string"""
        return self.xml_start() + self.xml_body() + self.xml_end()

    def xml_start(self):
        if not hasattr(self, 'inline'):
            return ['<%s>' % self.__class__.__name__]
        xmlns = 'http://www.w3.org/1998/Math/MathML'
        if self.inline:
            return ['<math xmlns="%s">' % xmlns]
        else:
            return ['<math xmlns="%s" mode="display">' % xmlns]

    def xml_end(self):
        return ['</%s>' % self.__class__.__name__]

    def xml_body(self):
        xml = []
        for child in self.children:
            xml.extend(child.xml())
        return xml