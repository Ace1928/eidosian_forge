import argparse
import os
from filelock import FileLock
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.datasets.mnist import load_data
from ray import train, tune
class MNISTTrainable(tune.Trainable):

    def setup(self, config):
        import tensorflow as tf
        with FileLock(os.path.expanduser('~/.tune.lock')):
            (x_train, y_train), (x_test, y_test) = load_data()
        x_train, x_test = (x_train / 255.0, x_test / 255.0)
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_ds = self.train_ds.shuffle(10000).batch(config.get('batch', 32))
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        self.model = MyModel(hiddens=config.get('hiddens', 128))
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images)
                loss = self.loss_object(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = self.model(images)
            t_loss = self.loss_object(labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(labels, predictions)
        self.tf_train_step = train_step
        self.tf_test_step = test_step

    def save_checkpoint(self, checkpoint_dir: str):
        return None

    def load_checkpoint(self, checkpoint):
        return None

    def step(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        for idx, (images, labels) in enumerate(self.train_ds):
            if idx > MAX_TRAIN_BATCH:
                break
            self.tf_train_step(images, labels)
        for test_images, test_labels in self.test_ds:
            self.tf_test_step(test_images, test_labels)
        return {'epoch': self.iteration, 'loss': self.train_loss.result().numpy(), 'accuracy': self.train_accuracy.result().numpy() * 100, 'test_loss': self.test_loss.result().numpy(), 'mean_accuracy': self.test_accuracy.result().numpy() * 100}