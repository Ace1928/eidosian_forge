import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def _create_training_examples_arc_eager(self, depgraphs, input_file):
    """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        """
    operation = Transition(self.ARC_EAGER)
    countProj = 0
    training_seq = []
    for depgraph in depgraphs:
        if not self._is_projective(depgraph):
            continue
        countProj += 1
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
                    key = Transition.RIGHT_ARC + ':' + rel
                    self._write_to_file(key, binary_features, input_file)
                    operation.right_arc(conf, rel)
                    training_seq.append(key)
                    continue
                flag = False
                for k in range(s0):
                    if self._get_dep_relation(k, b0, depgraph) is not None:
                        flag = True
                    if self._get_dep_relation(b0, k, depgraph) is not None:
                        flag = True
                if flag:
                    key = Transition.REDUCE
                    self._write_to_file(key, binary_features, input_file)
                    operation.reduce(conf)
                    training_seq.append(key)
                    continue
            key = Transition.SHIFT
            self._write_to_file(key, binary_features, input_file)
            operation.shift(conf)
            training_seq.append(key)
    print(' Number of training examples : ' + str(len(depgraphs)))
    print(' Number of valid (projective) examples : ' + str(countProj))
    return training_seq