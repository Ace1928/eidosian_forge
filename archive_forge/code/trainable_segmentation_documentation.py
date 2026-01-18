from skimage.feature import multiscale_basic_features
Segment new image using trained internal classifier.

        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.

        Raises
        ------
        NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
        