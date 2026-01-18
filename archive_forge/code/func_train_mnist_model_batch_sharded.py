import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src.datasets import mnist
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.utils import np_utils
def train_mnist_model_batch_sharded(model, optimizer, mesh, num_epochs, steps_per_epoch, global_batch_size):
    dataset, _ = get_mnist_datasets(NUM_CLASS, global_batch_size)
    input_image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
    input_label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
    loss_obj = losses.CategoricalCrossentropy()
    num_local_devices = mesh.num_local_devices()
    iterator = iter(dataset)
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            images, labels = next(iterator)
            images = tf.split(images, num_local_devices)
            labels = tf.split(labels, num_local_devices)
            d_images = dtensor.pack(images, input_image_layout)
            d_labels = dtensor.pack(labels, input_label_layout)
            total_loss += train_step(model, d_images, d_labels, loss_obj, optimizer)
        train_loss = tf.reduce_mean(total_loss / steps_per_epoch)
        logging.info('Epoch %d, Loss: %f', epoch, train_loss)
        train_losses.append(train_loss)
    return train_losses