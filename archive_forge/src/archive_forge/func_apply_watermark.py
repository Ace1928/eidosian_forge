import numpy as np
from imwatermark import WatermarkEncoder
def apply_watermark(self, images: np.array):
    if images.shape[-1] < 256:
        return images
    if images.dtype == np.float16:
        images = images.astype(np.float32)
    images = (255 * (images / 2 + 0.5)).transpose((0, 2, 3, 1))
    images = np.array([self.encoder.encode(image, 'dwtDct') for image in images]).transpose((0, 3, 1, 2))
    np.clip(2 * (images / 255 - 0.5), -1.0, 1.0, out=images)
    return images