from tensorboard.compat import tf2 as tf
from tensorboard.plugins.image import metadata
from tensorboard.util import lazy_tensor_creator
Write an image summary.

    See also `tf.summary.scalar`, `tf.summary.SummaryWriter`.

    Writes a collection of images to the current default summary writer. Data
    appears in TensorBoard's 'Images' dashboard. Like `tf.summary.scalar` points,
    each collection of images is associated with a `step` and a `name`.  All the
    image collections with the same `name` constitute a time series of image
    collections.

    This example writes 2 random grayscale images:

    ```python
    w = tf.summary.create_file_writer('test/logs')
    with w.as_default():
      image1 = tf.random.uniform(shape=[8, 8, 1])
      image2 = tf.random.uniform(shape=[8, 8, 1])
      tf.summary.image("grayscale_noise", [image1, image2], step=0)
    ```

    To avoid clipping, data should be converted to one of the following:

    - floating point values in the range [0,1], or
    - uint8 values in the range [0,255]

    ```python
    # Convert the original dtype=int32 `Tensor` into `dtype=float64`.
    rgb_image_float = tf.constant([
      [[1000, 0, 0], [0, 500, 1000]],
    ]) / 1000
    tf.summary.image("picture", [rgb_image_float], step=0)

    # Convert original dtype=uint8 `Tensor` into proper range.
    rgb_image_uint8 = tf.constant([
      [[1, 1, 0], [0, 0, 1]],
    ], dtype=tf.uint8) * 255
    tf.summary.image("picture", [rgb_image_uint8], step=1)
    ```

    Arguments:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A `Tensor` representing pixel data with shape `[k, h, w, c]`,
        where `k` is the number of images, `h` and `w` are the height and
        width of the images, and `c` is the number of channels, which
        should be 1, 2, 3, or 4 (grayscale, grayscale with alpha, RGB, RGBA).
        Any of the dimensions may be statically unknown (i.e., `None`).
        Floating point data will be clipped to the range [0,1]. Other data types
        will be clipped into an allowed range for safe casting to uint8, using
        `tf.image.convert_image_dtype`.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
        many images will be emitted at each step. When more than
        `max_outputs` many images are provided, the first `max_outputs` many
        images will be used and the rest silently discarded.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    