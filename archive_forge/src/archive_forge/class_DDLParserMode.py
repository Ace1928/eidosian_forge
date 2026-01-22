from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
class DDLParserMode:
    """A mode in DDLParser.

  A mode has one entering sequence, a list of exit sequences and one escape
  sequence. A mode could be:
    * skipping (e.x. comments), which skips the matched text.
    * non-skpping, (e.x. strings), which emits the matched text.
  """

    def __init__(self, parser, enter_seq, exit_seqs, escape_sequences, is_to_skip):
        self.parser_ = parser
        self.enter_seq_ = enter_seq
        self.exit_seqs_ = exit_seqs
        self.escape_sequences_ = escape_sequences
        self.is_to_skip_ = is_to_skip

    def TryEnter(self):
        """Trys to enter into the mode."""
        res = self.parser_.StartsWith(self.enter_seq_)
        if res:
            self.parser_.EmitBuffer()
            self.parser_.Advance(len(self.enter_seq_))
        return res

    def Exit(self):
        if self.is_to_skip_:
            self.parser_.SkipBuffer()
        else:
            self.parser_.EmitBuffer()
        self.parser_.ExitMode()

    def FindExitSeqence(self):
        """Finds a matching exit sequence."""
        for s in self.exit_seqs_:
            if self.parser_.StartsWith(s):
                return s
        return None

    def Process(self):
        """Process the ddl at the current parser index."""
        if self.escape_sequences_:
            for seq in self.escape_sequences_:
                if self.parser_.StartsWith(seq):
                    self.parser_.Advance(len(self.escape_sequences_))
                    return
        exit_seq = self.FindExitSeqence()
        if not exit_seq:
            self.parser_.Advance(1)
            return
        if not self.is_to_skip_:
            self.parser_.Advance(len(exit_seq))
        self.Exit()