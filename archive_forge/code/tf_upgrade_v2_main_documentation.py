import argparse
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import ipynb
from tensorflow.tools.compatibility import tf_upgrade_v2
from tensorflow.tools.compatibility import tf_upgrade_v2_safety
Process a file of type `.py` or `.ipynb`.