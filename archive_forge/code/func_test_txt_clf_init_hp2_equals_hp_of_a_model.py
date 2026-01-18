import copy
import keras_tuner
import autokeras as ak
from autokeras.tuners import task_specific
def test_txt_clf_init_hp2_equals_hp_of_a_model(tmp_path):
    clf = ak.TextClassifier(directory=tmp_path)
    clf.inputs[0].shape = (1,)
    clf.inputs[0].batch_size = 6
    clf.inputs[0].num_samples = 1000
    clf.outputs[0].in_blocks[0].shape = (10,)
    clf.tuner.hypermodel.epochs = 1000
    clf.tuner.hypermodel.num_samples = 20000
    init_hp = task_specific.TEXT_CLASSIFIER[2]
    hp = keras_tuner.HyperParameters()
    hp.values = copy.copy(init_hp)
    clf.tuner.hypermodel.build(hp)
    assert set(init_hp.keys()) == set(hp._hps.keys())