import re
from io import StringIO
from Bio.Phylo import Newick
def _parse_tree(self, text):
    """Parse the text representation into an Tree object (PRIVATE)."""
    tokens = re.finditer(tokenizer, text.strip())
    new_clade = self.new_clade
    root_clade = new_clade()
    current_clade = root_clade
    entering_branch_length = False
    lp_count = 0
    rp_count = 0
    for match in tokens:
        token = match.group()
        if token.startswith("'"):
            current_clade.name = token[1:-1]
        elif token.startswith('['):
            current_clade.comment = token[1:-1]
            if self.comments_are_confidence:
                current_clade.confidence = _parse_confidence(current_clade.comment)
        elif token == '(':
            current_clade = new_clade(current_clade)
            entering_branch_length = False
            lp_count += 1
        elif token == ',':
            if current_clade is root_clade:
                root_clade = new_clade()
                current_clade.parent = root_clade
            parent = self.process_clade(current_clade)
            current_clade = new_clade(parent)
            entering_branch_length = False
        elif token == ')':
            parent = self.process_clade(current_clade)
            if not parent:
                raise NewickError('Parenthesis mismatch.')
            current_clade = parent
            entering_branch_length = False
            rp_count += 1
        elif token == ';':
            break
        elif token.startswith(':'):
            value = float(token[1:])
            if self.values_are_confidence:
                current_clade.confidence = value
            else:
                current_clade.branch_length = value
        elif token == '\n':
            pass
        else:
            current_clade.name = token
    if lp_count != rp_count:
        raise NewickError(f'Mismatch, {lp_count} open vs {rp_count} close parentheses.')
    try:
        next_token = next(tokens)
        raise NewickError(f'Text after semicolon in Newick tree: {next_token.group()}')
    except StopIteration:
        pass
    self.process_clade(current_clade)
    self.process_clade(root_clade)
    return Newick.Tree(root=root_clade, rooted=self.rooted)