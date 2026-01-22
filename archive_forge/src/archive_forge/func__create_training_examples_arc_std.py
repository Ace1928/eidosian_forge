import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def _create_training_examples_arc_std(self, depgraphs, input_file):
    """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
    operation = Transition(self.ARC_STANDARD)
    count_proj = 0
    training_seq = []
    for depgraph in depgraphs:
        if not self._is_projective(depgraph):
            continue
        count_proj += 1
        conf = Configuration(depgraph)
        while len(conf.buffer) > 0:
            b0 = conf.buffer[0]
            features = conf.extract_features()
            binary_features = self._convert_to_binary_features(features)
            if len(conf.stack) > 0:
                s0 = conf.stack[len(conf.stack) - 1]
                rel = self._get_dep_relation(b0, s0, depgraph)
                if rel is not None:
                    key = Transition.LEFT_ARC + ':' + rel
                    self._write_to_file(key, binary_features, input_file)
                    operation.left_arc(conf, rel)
                    training_seq.append(key)
                    continue
                rel = self._get_dep_relation(s0, b0, depgraph)
                if rel is not None:
                    precondition = True
                    maxID = conf._max_address
                    for w in range(maxID + 1):
                        if w != b0:
                            relw = self._get_dep_relation(b0, w, depgraph)
                            if relw is not None:
                                if (b0, relw, w) not in conf.arcs:
                                    precondition = False
                    if precondition:
                        key = Transition.RIGHT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.right_arc(conf, rel)
                        training_seq.append(key)
                        continue
            key = Transition.SHIFT
            self._write_to_file(key, binary_features, input_file)
            operation.shift(conf)
            training_seq.append(key)
    print(' Number of training examples : ' + str(len(depgraphs)))
    print(' Number of valid (projective) examples : ' + str(count_proj))
    return training_seq