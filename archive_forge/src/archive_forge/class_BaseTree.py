from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
class BaseTree(Tree):
    """
    @brief A generic tree implementation with no payload.

    You must subclass to
    actually have any user data.  ANTLR v3 uses a list of children approach
    instead of the child-sibling approach in v2.  A flat tree (a list) is
    an empty node whose children represent the list.  An empty, but
    non-null node is called "nil".
    """

    def __init__(self, node=None):
        """
        Create a new node from an existing node does nothing for BaseTree
        as there are no fields other than the children list, which cannot
        be copied as the children are not considered part of this node.
        """
        Tree.__init__(self)
        self.children = []
        self.parent = None
        self.childIndex = 0

    def getChild(self, i):
        try:
            return self.children[i]
        except IndexError:
            return None

    def getChildren(self):
        """@brief Get the children internal List

        Note that if you directly mess with
        the list, do so at your own risk.
        """
        return self.children

    def getFirstChildWithType(self, treeType):
        for child in self.children:
            if child.getType() == treeType:
                return child
        return None

    def getChildCount(self):
        return len(self.children)

    def addChild(self, childTree):
        """Add t as child of this node.

        Warning: if t has no children, but child does
        and child isNil then this routine moves children to t via
        t.children = child.children; i.e., without copying the array.
        """
        if childTree is None:
            return
        if childTree.isNil():
            if self.children is childTree.children:
                raise ValueError('attempt to add child list to itself')
            for idx, child in enumerate(childTree.children):
                child.parent = self
                child.childIndex = len(self.children) + idx
            self.children += childTree.children
        else:
            self.children.append(childTree)
            childTree.parent = self
            childTree.childIndex = len(self.children) - 1

    def addChildren(self, children):
        """Add all elements of kids list as children of this node"""
        self.children += children

    def setChild(self, i, t):
        if t is None:
            return
        if t.isNil():
            raise ValueError("Can't set single child to a list")
        self.children[i] = t
        t.parent = self
        t.childIndex = i

    def deleteChild(self, i):
        killed = self.children[i]
        del self.children[i]
        for idx, child in enumerate(self.children[i:]):
            child.childIndex = i + idx
        return killed

    def replaceChildren(self, startChildIndex, stopChildIndex, newTree):
        """
        Delete children from start to stop and replace with t even if t is
        a list (nil-root tree).  num of children can increase or decrease.
        For huge child lists, inserting children can force walking rest of
        children to set their childindex; could be slow.
        """
        if startChildIndex >= len(self.children) or stopChildIndex >= len(self.children):
            raise IndexError('indexes invalid')
        replacingHowMany = stopChildIndex - startChildIndex + 1
        if newTree.isNil():
            newChildren = newTree.children
        else:
            newChildren = [newTree]
        replacingWithHowMany = len(newChildren)
        delta = replacingHowMany - replacingWithHowMany
        if delta == 0:
            for idx, child in enumerate(newChildren):
                self.children[idx + startChildIndex] = child
                child.parent = self
                child.childIndex = idx + startChildIndex
        else:
            del self.children[startChildIndex:stopChildIndex + 1]
            self.children[startChildIndex:startChildIndex] = newChildren
            self.freshenParentAndChildIndexes(startChildIndex)

    def isNil(self):
        return False

    def freshenParentAndChildIndexes(self, offset=0):
        for idx, child in enumerate(self.children[offset:]):
            child.childIndex = idx + offset
            child.parent = self

    def sanityCheckParentAndChildIndexes(self, parent=None, i=-1):
        if parent != self.parent:
            raise ValueError("parents don't match; expected %r found %r" % (parent, self.parent))
        if i != self.childIndex:
            raise ValueError("child indexes don't match; expected %d found %d" % (i, self.childIndex))
        for idx, child in enumerate(self.children):
            child.sanityCheckParentAndChildIndexes(self, idx)

    def getChildIndex(self):
        """BaseTree doesn't track child indexes."""
        return 0

    def setChildIndex(self, index):
        """BaseTree doesn't track child indexes."""
        pass

    def getParent(self):
        """BaseTree doesn't track parent pointers."""
        return None

    def setParent(self, t):
        """BaseTree doesn't track parent pointers."""
        pass

    def toStringTree(self):
        """Print out a whole tree not just a node"""
        if len(self.children) == 0:
            return self.toString()
        buf = []
        if not self.isNil():
            buf.append('(')
            buf.append(self.toString())
            buf.append(' ')
        for i, child in enumerate(self.children):
            if i > 0:
                buf.append(' ')
            buf.append(child.toStringTree())
        if not self.isNil():
            buf.append(')')
        return ''.join(buf)

    def getLine(self):
        return 0

    def getCharPositionInLine(self):
        return 0

    def toString(self):
        """Override to say how a node (not a tree) should look as text"""
        raise NotImplementedError