from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.core import exceptions
class DDLParser:
    """Parser for splitting ddl statements preserving GoogleSQL strings literals.

  DDLParse has a list of modes. If any mode is selected, control is given to the
  mode. If no mode is selected, the parser trys to enter the first mode that
  could it could enter. The parser handles splitting statements upon ';'.

  During parsing, a DDL has the following parts:
    * parts that has been processed: emitted or skipped.
    * followed by a buffer that has been matched by the current mode, which
      could be emitted or skipped by a mode. The start index of which is
      mode_start_index_.
    * followed by the next character indexed by next_index_, which could direct
      the parser to enter or exit a mode.
    * followed by the unprocessed character.

  DDLParser:
    * acts as a default mode.
    * provides utilities uesd by ParserMode to drive the parsing.
  """

    def __init__(self, ddl):
        self.ddl_ = ddl
        self.next_index_ = 0
        self.mode_ = None
        self.mode_start_index_ = 0
        self.modes_ = [self.SkippingMode('--', ['\n', '\r']), self.PreservingMode('"""', ['"""'], ['\\"', '\\\\']), self.PreservingMode("'''", ["'''"], ["\\'", '\\\\']), self.PreservingMode('"', ['"'], ['\\"', '\\\\']), self.PreservingMode("'", ["'"], ["\\'", '\\\\']), self.PreservingMode('`', ['`'], ['\\`', '\\\\'])]
        self.statements_ = []
        self.StartNewStatement()
        self.logger_ = logging.getLogger('SpannerDDLParser')

    def SkippingMode(self, enter_seq, exit_seqs):
        return DDLParserMode(self, enter_seq, exit_seqs, None, True)

    def PreservingMode(self, enter_seq, exit_seqs, escape_sequences):
        return DDLParserMode(self, enter_seq, exit_seqs, escape_sequences, False)

    def IsEof(self):
        return self.next_index_ == len(self.ddl_)

    def Advance(self, l):
        self.next_index_ += l

    def StartNewStatement(self):
        self.ddl_parts_ = []
        self.statements_.append(self.ddl_parts_)

    def EmitBuffer(self):
        if self.mode_start_index_ >= self.next_index_:
            return
        self.ddl_parts_.append(self.ddl_[self.mode_start_index_:self.next_index_])
        self.SkipBuffer()
        self.logger_.debug('emitted: %s', self.ddl_parts_[-1])

    def SkipBuffer(self):
        self.mode_start_index_ = self.next_index_

    def EnterMode(self, mode):
        self.logger_.debug('enter mode: %s at index: %d', mode.enter_seq_, self.next_index_)
        self.mode_ = mode

    def ExitMode(self):
        self.logger_.debug('exit mode: %s at index: %d', self.mode_.enter_seq_, self.next_index_)
        self.mode_ = None

    def StartsWith(self, s):
        return self.ddl_[self.next_index_:].startswith(s)

    def Process(self):
        """Process the DDL."""
        while not self.IsEof():
            if self.mode_:
                self.mode_.Process()
                continue
            if self.ddl_[self.next_index_] == ';':
                self.EmitBuffer()
                self.StartNewStatement()
                self.mode_start_index_ += 1
                self.Advance(1)
                continue
            for m in self.modes_:
                if m.TryEnter():
                    self.EnterMode(m)
                    break
            if not self.mode_:
                self.Advance(1)
        if self.mode_ is not None:
            m = self.mode_
            if not m.is_to_skip_:
                raise DDLSyntaxError('Unclosed %s start at index: %d, %s' % (m.enter_seq_, self.mode_start_index_, self.ddl_))
            self.mode_.Exit()
        else:
            self.EmitBuffer()
        self.logger_.debug('ddls: %s', self.statements_)
        res = [''.join(frags) for frags in self.statements_ if frags]
        if res and res[-1].isspace():
            return res[:-1]
        return res