import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_node_labels(self):
    """Test tokenization of labeled nodes."""
    self.assertEqual(tgrep.tgrep_tokenize('S < @SBJ < (@VP < (@VB $.. @OBJ))'), ['S', '<', '@SBJ', '<', '(', '@VP', '<', '(', '@VB', '$..', '@OBJ', ')', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('S < @SBJ=s < (@VP=v < (@VB $.. @OBJ))'), ['S', '<', '@SBJ', '=', 's', '<', '(', '@VP', '=', 'v', '<', '(', '@VB', '$..', '@OBJ', ')', ')'])