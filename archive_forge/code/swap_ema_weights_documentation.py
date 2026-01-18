from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
Swaps model weights and EMA weights before and after evaluation.

    This callbacks replaces the model's weight values with the values of
    the optimizer's EMA weights (the exponential moving average of the past
    model weights values, implementing "Polyak averaging") before model
    evaluation, and restores the previous weights after evaluation.

    The `SwapEMAWeights` callback is to be used in conjunction with
    an optimizer that sets `use_ema=True`.

    Note that the weights are swapped in-place in order to save memory.
    The behavior is undefined if you modify the EMA weights
    or model weights in other callbacks.

    Example:

    ```python
    # Remember to set `use_ema=True` in the optimizer
    optimizer = SGD(use_ema=True)
    model.compile(optimizer=optimizer, loss=..., metrics=...)

    # Metrics will be computed with EMA weights
    model.fit(X_train, Y_train, callbacks=[SwapEMAWeights()])

    # If you want to save model checkpoint with EMA weights, you can set
    # `swap_on_epoch=True` and place ModelCheckpoint after SwapEMAWeights.
    model.fit(
        X_train,
        Y_train,
        callbacks=[SwapEMAWeights(swap_on_epoch=True), ModelCheckpoint(...)]
    )
    ```

    Args:
        swap_on_epoch: whether to perform swapping at `on_epoch_begin()`
            and `on_epoch_end()`. This is useful if you want to use
            EMA weights for other callbacks such as `ModelCheckpoint`.
            Defaults to `False`.
    