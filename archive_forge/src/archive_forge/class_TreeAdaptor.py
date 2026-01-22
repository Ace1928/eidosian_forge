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
class TreeAdaptor(object):
    """
    @brief Abstract baseclass for tree adaptors.

    How to create and navigate trees.  Rather than have a separate factory
    and adaptor, I've merged them.  Makes sense to encapsulate.

    This takes the place of the tree construction code generated in the
    generated code in 2.x and the ASTFactory.

    I do not need to know the type of a tree at all so they are all
    generic Objects.  This may increase the amount of typecasting needed. :(
    """

    def createWithPayload(self, payload):
        """
        Create a tree node from Token object; for CommonTree type trees,
        then the token just becomes the payload.  This is the most
        common create call.

        Override if you want another kind of node to be built.
        """
        raise NotImplementedError

    def dupNode(self, treeNode):
        """Duplicate a single tree node.

        Override if you want another kind of node to be built.
    """
        raise NotImplementedError

    def dupTree(self, tree):
        """Duplicate tree recursively, using dupNode() for each node"""
        raise NotImplementedError

    def nil(self):
        """
        Return a nil node (an empty but non-null node) that can hold
        a list of element as the children.  If you want a flat tree (a list)
        use "t=adaptor.nil(); t.addChild(x); t.addChild(y);"
        """
        raise NotImplementedError

    def errorNode(self, input, start, stop, exc):
        """
        Return a tree node representing an error.  This node records the
        tokens consumed during error recovery.  The start token indicates the
        input symbol at which the error was detected.  The stop token indicates
        the last symbol consumed during recovery.

        You must specify the input stream so that the erroneous text can
        be packaged up in the error node.  The exception could be useful
        to some applications; default implementation stores ptr to it in
        the CommonErrorNode.

        This only makes sense during token parsing, not tree parsing.
        Tree parsing should happen only when parsing and tree construction
        succeed.
        """
        raise NotImplementedError

    def isNil(self, tree):
        """Is tree considered a nil node used to make lists of child nodes?"""
        raise NotImplementedError

    def addChild(self, t, child):
        """
        Add a child to the tree t.  If child is a flat tree (a list), make all
        in list children of t.  Warning: if t has no children, but child does
        and child isNil then you can decide it is ok to move children to t via
        t.children = child.children; i.e., without copying the array.  Just
        make sure that this is consistent with have the user will build
        ASTs. Do nothing if t or child is null.
        """
        raise NotImplementedError

    def becomeRoot(self, newRoot, oldRoot):
        """
        If oldRoot is a nil root, just copy or move the children to newRoot.
        If not a nil root, make oldRoot a child of newRoot.

           old=^(nil a b c), new=r yields ^(r a b c)
           old=^(a b c), new=r yields ^(r ^(a b c))

        If newRoot is a nil-rooted single child tree, use the single
        child as the new root node.

           old=^(nil a b c), new=^(nil r) yields ^(r a b c)
           old=^(a b c), new=^(nil r) yields ^(r ^(a b c))

        If oldRoot was null, it's ok, just return newRoot (even if isNil).

           old=null, new=r yields r
           old=null, new=^(nil r) yields ^(nil r)

        Return newRoot.  Throw an exception if newRoot is not a
        simple node or nil root with a single child node--it must be a root
        node.  If newRoot is ^(nil x) return x as newRoot.

        Be advised that it's ok for newRoot to point at oldRoot's
        children; i.e., you don't have to copy the list.  We are
        constructing these nodes so we should have this control for
        efficiency.
        """
        raise NotImplementedError

    def rulePostProcessing(self, root):
        """
        Given the root of the subtree created for this rule, post process
        it to do any simplifications or whatever you want.  A required
        behavior is to convert ^(nil singleSubtree) to singleSubtree
        as the setting of start/stop indexes relies on a single non-nil root
        for non-flat trees.

        Flat trees such as for lists like "idlist : ID+ ;" are left alone
        unless there is only one ID.  For a list, the start/stop indexes
        are set in the nil node.

        This method is executed after all rule tree construction and right
        before setTokenBoundaries().
        """
        raise NotImplementedError

    def getUniqueID(self, node):
        """For identifying trees.

        How to identify nodes so we can say "add node to a prior node"?
        Even becomeRoot is an issue.  Use System.identityHashCode(node)
        usually.
        """
        raise NotImplementedError

    def createFromToken(self, tokenType, fromToken, text=None):
        """
        Create a new node derived from a token, with a new token type and
        (optionally) new text.

        This is invoked from an imaginary node ref on right side of a
        rewrite rule as IMAG[$tokenLabel] or IMAG[$tokenLabel "IMAG"].

        This should invoke createToken(Token).
        """
        raise NotImplementedError

    def createFromType(self, tokenType, text):
        """Create a new node derived from a token, with a new token type.

        This is invoked from an imaginary node ref on right side of a
        rewrite rule as IMAG["IMAG"].

        This should invoke createToken(int,String).
        """
        raise NotImplementedError

    def getType(self, t):
        """For tree parsing, I need to know the token type of a node"""
        raise NotImplementedError

    def setType(self, t, type):
        """Node constructors can set the type of a node"""
        raise NotImplementedError

    def getText(self, t):
        raise NotImplementedError

    def setText(self, t, text):
        """Node constructors can set the text of a node"""
        raise NotImplementedError

    def getToken(self, t):
        """Return the token object from which this node was created.

        Currently used only for printing an error message.
        The error display routine in BaseRecognizer needs to
        display where the input the error occurred. If your
        tree of limitation does not store information that can
        lead you to the token, you can create a token filled with
        the appropriate information and pass that back.  See
        BaseRecognizer.getErrorMessage().
        """
        raise NotImplementedError

    def setTokenBoundaries(self, t, startToken, stopToken):
        """
        Where are the bounds in the input token stream for this node and
        all children?  Each rule that creates AST nodes will call this
        method right before returning.  Flat trees (i.e., lists) will
        still usually have a nil root node just to hold the children list.
        That node would contain the start/stop indexes then.
        """
        raise NotImplementedError

    def getTokenStartIndex(self, t):
        """
        Get the token start index for this subtree; return -1 if no such index
        """
        raise NotImplementedError

    def getTokenStopIndex(self, t):
        """
        Get the token stop index for this subtree; return -1 if no such index
        """
        raise NotImplementedError

    def getChild(self, t, i):
        """Get a child 0..n-1 node"""
        raise NotImplementedError

    def setChild(self, t, i, child):
        """Set ith child (0..n-1) to t; t must be non-null and non-nil node"""
        raise NotImplementedError

    def deleteChild(self, t, i):
        """Remove ith child and shift children down from right."""
        raise NotImplementedError

    def getChildCount(self, t):
        """How many children?  If 0, then this is a leaf node"""
        raise NotImplementedError

    def getParent(self, t):
        """
        Who is the parent node of this node; if null, implies node is root.
        If your node type doesn't handle this, it's ok but the tree rewrites
        in tree parsers need this functionality.
        """
        raise NotImplementedError

    def setParent(self, t, parent):
        """
        Who is the parent node of this node; if null, implies node is root.
        If your node type doesn't handle this, it's ok but the tree rewrites
        in tree parsers need this functionality.
        """
        raise NotImplementedError

    def getChildIndex(self, t):
        """
        What index is this node in the child list? Range: 0..n-1
        If your node type doesn't handle this, it's ok but the tree rewrites
        in tree parsers need this functionality.
        """
        raise NotImplementedError

    def setChildIndex(self, t, index):
        """
        What index is this node in the child list? Range: 0..n-1
        If your node type doesn't handle this, it's ok but the tree rewrites
        in tree parsers need this functionality.
        """
        raise NotImplementedError

    def replaceChildren(self, parent, startChildIndex, stopChildIndex, t):
        """
        Replace from start to stop child index of parent with t, which might
        be a list.  Number of children may be different
        after this call.

        If parent is null, don't do anything; must be at root of overall tree.
        Can't replace whatever points to the parent externally.  Do nothing.
        """
        raise NotImplementedError

    def create(self, *args):
        """
        Deprecated, use createWithPayload, createFromToken or createFromType.

        This method only exists to mimic the Java interface of TreeAdaptor.

        """
        if len(args) == 1 and isinstance(args[0], Token):
            return self.createWithPayload(args[0])
        if len(args) == 2 and isinstance(args[0], six.integer_types) and isinstance(args[1], Token):
            return self.createFromToken(args[0], args[1])
        if len(args) == 3 and isinstance(args[0], six.integer_types) and isinstance(args[1], Token) and isinstance(args[2], six.string_types):
            return self.createFromToken(args[0], args[1], args[2])
        if len(args) == 2 and isinstance(args[0], six.integer_types) and isinstance(args[1], six.string_types):
            return self.createFromType(args[0], args[1])
        raise TypeError('No create method with this signature found: %s' % ', '.join((type(v).__name__ for v in args)))