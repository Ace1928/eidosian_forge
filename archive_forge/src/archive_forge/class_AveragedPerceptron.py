import logging
import pickle
import random
from collections import defaultdict
from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI
@jsontags.register_tag
class AveragedPerceptron:
    """An averaged perceptron, as implemented by Matthew Honnibal.

    See more implementation details here:
        https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
    """
    json_tag = 'nltk.tag.perceptron.AveragedPerceptron'

    def __init__(self, weights=None):
        self.weights = weights if weights else {}
        self.classes = set()
        self._totals = defaultdict(int)
        self._tstamps = defaultdict(int)
        self.i = 0

    def _softmax(self, scores):
        s = np.fromiter(scores.values(), dtype=float)
        exps = np.exp(s)
        return exps / np.sum(exps)

    def predict(self, features, return_conf=False):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        best_label = max(self.classes, key=lambda label: (scores[label], label))
        conf = max(self._softmax(scores)) if return_conf == True else None
        return (best_label, conf)

    def update(self, truth, guess, features):
        """Update the feature weights."""

        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v
        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)

    def average_weights(self):
        """Average weights from all iterations."""
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / self.i, 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def save(self, path):
        """Save the pickled model weights."""
        with open(path, 'wb') as fout:
            return pickle.dump(dict(self.weights), fout)

    def load(self, path):
        """Load the pickled model weights."""
        self.weights = load(path)

    def encode_json_obj(self):
        return self.weights

    @classmethod
    def decode_json_obj(cls, obj):
        return cls(obj)