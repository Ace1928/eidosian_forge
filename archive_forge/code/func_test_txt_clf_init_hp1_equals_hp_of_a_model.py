import copy
import keras_tuner
import autokeras as ak
from autokeras.tuners import task_specific
def test_txt_clf_init_hp1_equals_hp_of_a_model(tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.outputs[0].in_blocks[0].shape = (10,)
    init_hp = task_specific.TEXT_CLASSIFIER[1]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())