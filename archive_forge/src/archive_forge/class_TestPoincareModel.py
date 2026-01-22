import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
class TestPoincareModel(unittest.TestCase):

    def setUp(self):
        self.data = PoincareRelations(datapath('poincare_hypernyms.tsv'))
        self.data_large = PoincareRelations(datapath('poincare_hypernyms_large.tsv'))

    def models_equal(self, model_1, model_2):
        self.assertEqual(len(model_1.kv), len(model_2.kv))
        self.assertEqual(set(model_1.kv.index_to_key), set(model_2.kv.index_to_key))
        self.assertTrue(np.allclose(model_1.kv.vectors, model_2.kv.vectors))

    def test_data_counts(self):
        """Tests whether data has been loaded correctly and completely."""
        model = PoincareModel(self.data)
        self.assertEqual(len(model.all_relations), 5)
        self.assertEqual(len(model.node_relations[model.kv.get_index('kangaroo.n.01')]), 3)
        self.assertEqual(len(model.kv), 7)
        self.assertTrue('mammal.n.01' not in model.node_relations)

    def test_data_counts_with_bytes(self):
        """Tests whether input bytes data is loaded correctly and completely."""
        model = PoincareModel([(b'\x80\x01c', b'Pqa'), (b'node.1', b'node.2')])
        self.assertEqual(len(model.all_relations), 2)
        self.assertEqual(len(model.node_relations[model.kv.get_index(b'\x80\x01c')]), 1)
        self.assertEqual(len(model.kv), 4)
        self.assertTrue(b'Pqa' not in model.node_relations)

    def test_persistence(self):
        """Tests whether the model is saved and loaded correctly."""
        model = PoincareModel(self.data, burn_in=0, negative=3)
        model.train(epochs=1)
        model.save(testfile())
        loaded = PoincareModel.load(testfile())
        self.models_equal(model, loaded)

    def test_persistence_separate_file(self):
        """Tests whether the model is saved and loaded correctly when the arrays are stored separately."""
        model = PoincareModel(self.data, burn_in=0, negative=3)
        model.train(epochs=1)
        model.save(testfile(), sep_limit=1)
        loaded = PoincareModel.load(testfile())
        self.models_equal(model, loaded)

    def test_online_learning(self):
        """Tests whether additional input data is loaded correctly and completely."""
        model = PoincareModel(self.data, burn_in=0, negative=3)
        self.assertEqual(len(model.kv), 7)
        self.assertEqual(model.kv.get_vecattr('kangaroo.n.01', 'count'), 3)
        self.assertEqual(model.kv.get_vecattr('cat.n.01', 'count'), 1)
        model.build_vocab([('kangaroo.n.01', 'cat.n.01')], update=True)
        self.assertEqual(model.kv.get_vecattr('kangaroo.n.01', 'count'), 4)
        self.assertEqual(model.kv.get_vecattr('cat.n.01', 'count'), 2)

    def test_train_after_load(self):
        """Tests whether the model can be trained correctly after loading from disk."""
        model = PoincareModel(self.data, burn_in=0, negative=3)
        model.train(epochs=1)
        model.save(testfile())
        loaded = PoincareModel.load(testfile())
        model.train(epochs=1)
        loaded.train(epochs=1)
        self.models_equal(model, loaded)

    def test_persistence_old_model(self):
        """Tests whether model from older gensim version is loaded correctly."""
        loaded = PoincareModel.load(datapath('poincare_test_3.4.0'))
        self.assertEqual(loaded.kv.vectors.shape, (239, 2))
        self.assertEqual(len(loaded.kv), 239)
        self.assertEqual(loaded.size, 2)
        self.assertEqual(len(loaded.all_relations), 200)

    def test_train_old_model_after_load(self):
        """Tests whether loaded model from older gensim version can be trained correctly."""
        loaded = PoincareModel.load(datapath('poincare_test_3.4.0'))
        old_vectors = np.copy(loaded.kv.vectors)
        loaded.train(epochs=2)
        self.assertFalse(np.allclose(old_vectors, loaded.kv.vectors))

    def test_invalid_data_raises_error(self):
        """Tests that error is raised on invalid input data."""
        with self.assertRaises(ValueError):
            PoincareModel([('a', 'b', 'c')])
        with self.assertRaises(ValueError):
            PoincareModel(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            PoincareModel('ab')

    def test_vector_shape(self):
        """Tests whether vectors are initialized with the correct size."""
        model = PoincareModel(self.data, size=20)
        self.assertEqual(model.kv.vectors.shape, (7, 20))

    def test_vector_dtype(self):
        """Tests whether vectors have the correct dtype before and after training."""
        model = PoincareModel(self.data_large, dtype=np.float32, burn_in=0, negative=3)
        self.assertEqual(model.kv.vectors.dtype, np.float32)
        model.train(epochs=1)
        self.assertEqual(model.kv.vectors.dtype, np.float32)

    def test_training(self):
        """Tests that vectors are different before and after training."""
        model = PoincareModel(self.data_large, burn_in=0, negative=3)
        old_vectors = np.copy(model.kv.vectors)
        model.train(epochs=2)
        self.assertFalse(np.allclose(old_vectors, model.kv.vectors))

    def test_training_multiple(self):
        """Tests that calling train multiple times results in different vectors."""
        model = PoincareModel(self.data_large, burn_in=0, negative=3)
        model.train(epochs=2)
        old_vectors = np.copy(model.kv.vectors)
        model.train(epochs=1)
        self.assertFalse(np.allclose(old_vectors, model.kv.vectors))
        old_vectors = np.copy(model.kv.vectors)
        model.train(epochs=0)
        self.assertTrue(np.allclose(old_vectors, model.kv.vectors))

    def test_gradients_check(self):
        """Tests that the model is trained successfully with gradients check enabled."""
        model = PoincareModel(self.data, negative=3)
        try:
            model.train(epochs=1, batch_size=1, check_gradients_every=1)
        except Exception as e:
            self.fail('Exception %s raised unexpectedly while training with gradient checking' % repr(e))

    @unittest.skipIf(not autograd_installed, 'autograd needs to be installed for this test')
    def test_wrong_gradients_raises_assertion(self):
        """Tests that discrepancy in gradients raises an error."""
        model = PoincareModel(self.data, negative=3)
        model._loss_grad = Mock(return_value=np.zeros((2 + model.negative, model.size)))
        with self.assertRaises(AssertionError):
            model.train(epochs=1, batch_size=1, check_gradients_every=1)

    def test_reproducible(self):
        """Tests that vectors are same for two independent models trained with the same seed."""
        model_1 = PoincareModel(self.data_large, seed=1, negative=3, burn_in=1)
        model_1.train(epochs=2)
        model_2 = PoincareModel(self.data_large, seed=1, negative=3, burn_in=1)
        model_2.train(epochs=2)
        self.assertTrue(np.allclose(model_1.kv.vectors, model_2.kv.vectors))

    def test_burn_in(self):
        """Tests that vectors are different after burn-in."""
        model = PoincareModel(self.data, burn_in=1, negative=3)
        original_vectors = np.copy(model.kv.vectors)
        model.train(epochs=0)
        self.assertFalse(np.allclose(model.kv.vectors, original_vectors))

    def test_burn_in_only_done_once(self):
        """Tests that burn-in does not happen when train is called a second time."""
        model = PoincareModel(self.data, negative=3, burn_in=1)
        model.train(epochs=0)
        original_vectors = np.copy(model.kv.vectors)
        model.train(epochs=0)
        self.assertTrue(np.allclose(model.kv.vectors, original_vectors))

    def test_negatives(self):
        """Tests that correct number of negatives are sampled."""
        model = PoincareModel(self.data, negative=5)
        self.assertEqual(len(model._get_candidate_negatives()), 5)

    def test_error_if_negative_more_than_population(self):
        """Tests error is rased if number of negatives to sample is more than remaining nodes."""
        model = PoincareModel(self.data, negative=5)
        with self.assertRaises(ValueError):
            model.train(epochs=1)

    def test_no_duplicates_and_positives_in_negative_sample(self):
        """Tests that no duplicates or positively related nodes are present in negative samples."""
        model = PoincareModel(self.data_large, negative=3)
        positive_nodes = model.node_relations[0]
        num_samples = 100
        for i in range(num_samples):
            negatives = model._sample_negatives(0)
            self.assertFalse(positive_nodes & set(negatives))
            self.assertEqual(len(negatives), len(set(negatives)))

    def test_handle_duplicates(self):
        """Tests that correct number of negatives are used."""
        vector_updates = np.array([[0.5, 0.5], [0.1, 0.2], [0.3, -0.2]])
        node_indices = [0, 1, 0]
        PoincareModel._handle_duplicates(vector_updates, node_indices)
        vector_updates_expected = np.array([[0.0, 0.0], [0.1, 0.2], [0.8, 0.3]])
        self.assertTrue((vector_updates == vector_updates_expected).all())

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(testfile())
        except OSError:
            pass