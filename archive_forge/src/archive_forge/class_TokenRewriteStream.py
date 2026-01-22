from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class TokenRewriteStream(CommonTokenStream):
    """@brief CommonTokenStream that can be modified.

    Useful for dumping out the input stream after doing some
    augmentation or other manipulations.

    You can insert stuff, replace, and delete chunks.  Note that the
    operations are done lazily--only if you convert the buffer to a
    String.  This is very efficient because you are not moving data around
    all the time.  As the buffer of tokens is converted to strings, the
    toString() method(s) check to see if there is an operation at the
    current index.  If so, the operation is done and then normal String
    rendering continues on the buffer.  This is like having multiple Turing
    machine instruction streams (programs) operating on a single input tape. :)

    Since the operations are done lazily at toString-time, operations do not
    screw up the token index values.  That is, an insert operation at token
    index i does not change the index values for tokens i+1..n-1.

    Because operations never actually alter the buffer, you may always get
    the original token stream back without undoing anything.  Since
    the instructions are queued up, you can easily simulate transactions and
    roll back any changes if there is an error just by removing instructions.
    For example,

     CharStream input = new ANTLRFileStream("input");
     TLexer lex = new TLexer(input);
     TokenRewriteStream tokens = new TokenRewriteStream(lex);
     T parser = new T(tokens);
     parser.startRule();

     Then in the rules, you can execute
        Token t,u;
        ...
        input.insertAfter(t, "text to put after t");}
        input.insertAfter(u, "text after u");}
        System.out.println(tokens.toString());

    Actually, you have to cast the 'input' to a TokenRewriteStream. :(

    You can also have multiple "instruction streams" and get multiple
    rewrites from a single pass over the input.  Just name the instruction
    streams and use that name again when printing the buffer.  This could be
    useful for generating a C file and also its header file--all from the
    same buffer:

        tokens.insertAfter("pass1", t, "text to put after t");}
        tokens.insertAfter("pass2", u, "text after u");}
        System.out.println(tokens.toString("pass1"));
        System.out.println(tokens.toString("pass2"));

    If you don't use named rewrite streams, a "default" stream is used as
    the first example shows.
    """
    DEFAULT_PROGRAM_NAME = 'default'
    MIN_TOKEN_INDEX = 0

    def __init__(self, tokenSource=None, channel=DEFAULT_CHANNEL):
        CommonTokenStream.__init__(self, tokenSource, channel)
        self.programs = {}
        self.programs[self.DEFAULT_PROGRAM_NAME] = []
        self.lastRewriteTokenIndexes = {}

    def rollback(self, *args):
        """
        Rollback the instruction stream for a program so that
        the indicated instruction (via instructionIndex) is no
        longer in the stream.  UNTESTED!
        """
        if len(args) == 2:
            programName = args[0]
            instructionIndex = args[1]
        elif len(args) == 1:
            programName = self.DEFAULT_PROGRAM_NAME
            instructionIndex = args[0]
        else:
            raise TypeError('Invalid arguments')
        p = self.programs.get(programName, None)
        if p is not None:
            self.programs[programName] = p[self.MIN_TOKEN_INDEX:instructionIndex]

    def deleteProgram(self, programName=DEFAULT_PROGRAM_NAME):
        """Reset the program so that no instructions exist"""
        self.rollback(programName, self.MIN_TOKEN_INDEX)

    def insertAfter(self, *args):
        if len(args) == 2:
            programName = self.DEFAULT_PROGRAM_NAME
            index = args[0]
            text = args[1]
        elif len(args) == 3:
            programName = args[0]
            index = args[1]
            text = args[2]
        else:
            raise TypeError('Invalid arguments')
        if isinstance(index, Token):
            index = index.index
        self.insertBefore(programName, index + 1, text)

    def insertBefore(self, *args):
        if len(args) == 2:
            programName = self.DEFAULT_PROGRAM_NAME
            index = args[0]
            text = args[1]
        elif len(args) == 3:
            programName = args[0]
            index = args[1]
            text = args[2]
        else:
            raise TypeError('Invalid arguments')
        if isinstance(index, Token):
            index = index.index
        op = InsertBeforeOp(self, index, text)
        rewrites = self.getProgram(programName)
        rewrites.append(op)

    def replace(self, *args):
        if len(args) == 2:
            programName = self.DEFAULT_PROGRAM_NAME
            first = args[0]
            last = args[0]
            text = args[1]
        elif len(args) == 3:
            programName = self.DEFAULT_PROGRAM_NAME
            first = args[0]
            last = args[1]
            text = args[2]
        elif len(args) == 4:
            programName = args[0]
            first = args[1]
            last = args[2]
            text = args[3]
        else:
            raise TypeError('Invalid arguments')
        if isinstance(first, Token):
            first = first.index
        if isinstance(last, Token):
            last = last.index
        if first > last or first < 0 or last < 0 or (last >= len(self.tokens)):
            raise ValueError('replace: range invalid: ' + first + '..' + last + '(size=' + len(self.tokens) + ')')
        op = ReplaceOp(self, first, last, text)
        rewrites = self.getProgram(programName)
        rewrites.append(op)

    def delete(self, *args):
        self.replace(*list(args) + [None])

    def getLastRewriteTokenIndex(self, programName=DEFAULT_PROGRAM_NAME):
        return self.lastRewriteTokenIndexes.get(programName, -1)

    def setLastRewriteTokenIndex(self, programName, i):
        self.lastRewriteTokenIndexes[programName] = i

    def getProgram(self, name):
        p = self.programs.get(name, None)
        if p is None:
            p = self.initializeProgram(name)
        return p

    def initializeProgram(self, name):
        p = []
        self.programs[name] = p
        return p

    def toOriginalString(self, start=None, end=None):
        if start is None:
            start = self.MIN_TOKEN_INDEX
        if end is None:
            end = self.size() - 1
        buf = StringIO()
        i = start
        while i >= self.MIN_TOKEN_INDEX and i <= end and (i < len(self.tokens)):
            buf.write(self.get(i).text)
            i += 1
        return buf.getvalue()

    def toString(self, *args):
        if len(args) == 0:
            programName = self.DEFAULT_PROGRAM_NAME
            start = self.MIN_TOKEN_INDEX
            end = self.size() - 1
        elif len(args) == 1:
            programName = args[0]
            start = self.MIN_TOKEN_INDEX
            end = self.size() - 1
        elif len(args) == 2:
            programName = self.DEFAULT_PROGRAM_NAME
            start = args[0]
            end = args[1]
        if start is None:
            start = self.MIN_TOKEN_INDEX
        elif not isinstance(start, int):
            start = start.index
        if end is None:
            end = len(self.tokens) - 1
        elif not isinstance(end, int):
            end = end.index
        if end >= len(self.tokens):
            end = len(self.tokens) - 1
        if start < 0:
            start = 0
        rewrites = self.programs.get(programName)
        if rewrites is None or len(rewrites) == 0:
            return self.toOriginalString(start, end)
        buf = StringIO()
        indexToOp = self.reduceToSingleOperationPerIndex(rewrites)
        i = start
        while i <= end and i < len(self.tokens):
            op = indexToOp.get(i)
            try:
                del indexToOp[i]
            except KeyError:
                pass
            t = self.tokens[i]
            if op is None:
                buf.write(t.text)
                i += 1
            else:
                i = op.execute(buf)
        if end == len(self.tokens) - 1:
            for i in sorted(indexToOp.keys()):
                op = indexToOp[i]
                if op.index >= len(self.tokens) - 1:
                    buf.write(op.text)
        return buf.getvalue()
    __str__ = toString

    def reduceToSingleOperationPerIndex(self, rewrites):
        """
        We need to combine operations and report invalid operations (like
        overlapping replaces that are not completed nested).  Inserts to
        same index need to be combined etc...   Here are the cases:

        I.i.u I.j.v                           leave alone, nonoverlapping
        I.i.u I.i.v                           combine: Iivu

        R.i-j.u R.x-y.v | i-j in x-y          delete first R
        R.i-j.u R.i-j.v                       delete first R
        R.i-j.u R.x-y.v | x-y in i-j          ERROR
        R.i-j.u R.x-y.v | boundaries overlap  ERROR

        I.i.u R.x-y.v   | i in x-y            delete I
        I.i.u R.x-y.v   | i not in x-y        leave alone, nonoverlapping
        R.x-y.v I.i.u   | i in x-y            ERROR
        R.x-y.v I.x.u                         R.x-y.uv (combine, delete I)
        R.x-y.v I.i.u   | i not in x-y        leave alone, nonoverlapping

        I.i.u = insert u before op @ index i
        R.x-y.u = replace x-y indexed tokens with u

        First we need to examine replaces.  For any replace op:

          1. wipe out any insertions before op within that range.
          2. Drop any replace op before that is contained completely within
             that range.
          3. Throw exception upon boundary overlap with any previous replace.

        Then we can deal with inserts:

          1. for any inserts to same index, combine even if not adjacent.
          2. for any prior replace with same left boundary, combine this
             insert with replace and delete this replace.
          3. throw exception if index in same range as previous replace

        Don't actually delete; make op null in list. Easier to walk list.
        Later we can throw as we add to index -> op map.

        Note that I.2 R.2-2 will wipe out I.2 even though, technically, the
        inserted stuff would be before the replace range.  But, if you
        add tokens in front of a method body '{' and then delete the method
        body, I think the stuff before the '{' you added should disappear too.

        Return a map from token index to operation.
        """
        for i, rop in enumerate(rewrites):
            if rop is None:
                continue
            if not isinstance(rop, ReplaceOp):
                continue
            for j, iop in self.getKindOfOps(rewrites, InsertBeforeOp, i):
                if iop.index >= rop.index and iop.index <= rop.lastIndex:
                    rewrites[j] = None
            for j, prevRop in self.getKindOfOps(rewrites, ReplaceOp, i):
                if prevRop.index >= rop.index and prevRop.lastIndex <= rop.lastIndex:
                    rewrites[j] = None
                    continue
                disjoint = prevRop.lastIndex < rop.index or prevRop.index > rop.lastIndex
                same = prevRop.index == rop.index and prevRop.lastIndex == rop.lastIndex
                if not disjoint and (not same):
                    raise ValueError('replace op boundaries of %s overlap with previous %s' % (rop, prevRop))
        for i, iop in enumerate(rewrites):
            if iop is None:
                continue
            if not isinstance(iop, InsertBeforeOp):
                continue
            for j, prevIop in self.getKindOfOps(rewrites, InsertBeforeOp, i):
                if prevIop.index == iop.index:
                    iop.text = self.catOpText(iop.text, prevIop.text)
                    rewrites[j] = None
            for j, rop in self.getKindOfOps(rewrites, ReplaceOp, i):
                if iop.index == rop.index:
                    rop.text = self.catOpText(iop.text, rop.text)
                    rewrites[i] = None
                    continue
                if iop.index >= rop.index and iop.index <= rop.lastIndex:
                    raise ValueError('insert op %s within boundaries of previous %s' % (iop, rop))
        m = {}
        for i, op in enumerate(rewrites):
            if op is None:
                continue
            assert op.index not in m, 'should only be one op per index'
            m[op.index] = op
        return m

    def catOpText(self, a, b):
        x = ''
        y = ''
        if a is not None:
            x = a
        if b is not None:
            y = b
        return x + y

    def getKindOfOps(self, rewrites, kind, before=None):
        if before is None:
            before = len(rewrites)
        elif before > len(rewrites):
            before = len(rewrites)
        for i, op in enumerate(rewrites[:before]):
            if op is None:
                continue
            if op.__class__ == kind:
                yield (i, op)

    def toDebugString(self, start=None, end=None):
        if start is None:
            start = self.MIN_TOKEN_INDEX
        if end is None:
            end = self.size() - 1
        buf = StringIO()
        i = start
        while i >= self.MIN_TOKEN_INDEX and i <= end and (i < len(self.tokens)):
            buf.write(self.get(i))
            i += 1
        return buf.getvalue()