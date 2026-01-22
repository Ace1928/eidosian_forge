from docutils import nodes, languages
from docutils.transforms import parts
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
class Sectnum(Directive):
    """Automatic section numbering."""
    option_spec = {'depth': int, 'start': int, 'prefix': directives.unchanged_required, 'suffix': directives.unchanged_required}

    def run(self):
        pending = nodes.pending(parts.SectNum)
        pending.details.update(self.options)
        self.state_machine.document.note_pending(pending)
        return [pending]