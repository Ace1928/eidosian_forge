import numpy as np
from scipy.signal import convolve
from .._shared.utils import _supported_float_type
from . import uft
def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True, clip=True, *, rng=None):
    """Unsupervised Wiener-Hunt deconvolution.

    Return the deconvolution with a Wiener-Hunt approach, where the
    hyperparameters are automatically estimated. The algorithm is a
    stochastic iterative process (Gibbs sampler) described in the
    reference below. See also ``wiener`` function.

    Parameters
    ----------
    image : (M, N) ndarray
       The input degraded image.
    psf : ndarray
       The impulse response (input image's space) or the transfer
       function (Fourier space). Both are accepted. The transfer
       function is automatically recognized as being complex
       (``np.iscomplexobj(psf)``).
    reg : ndarray, optional
       The regularisation operator. The Laplacian by default. It can
       be an impulse response or a transfer function, as for the psf.
    user_params : dict, optional
       Dictionary of parameters for the Gibbs sampler. See below.
    clip : boolean, optional
       True by default. If true, pixel values of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

        .. versionadded:: 0.19

    Returns
    -------
    x_postmean : (M, N) ndarray
       The deconvolved image (the posterior mean).
    chains : dict
       The keys ``noise`` and ``prior`` contain the chain list of
       noise and prior precision respectively.

    Other parameters
    ----------------
    The keys of ``user_params`` are:

    threshold : float
       The stopping criterion: the norm of the difference between to
       successive approximated solution (empirical mean of object
       samples, see Notes section). 1e-4 by default.
    burnin : int
       The number of sample to ignore to start computation of the
       mean. 15 by default.
    min_num_iter : int
       The minimum number of iterations. 30 by default.
    max_num_iter : int
       The maximum number of iterations if ``threshold`` is not
       satisfied. 200 by default.
    callback : callable (None by default)
       A user provided callable to which is passed, if the function
       exists, the current image sample for whatever purpose. The user
       can store the sample, or compute other moments than the
       mean. It has no influence on the algorithm execution and is
       only for inspection.

    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> img = color.rgb2gray(data.astronaut())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> img = convolve2d(img, psf, 'same')
    >>> rng = np.random.default_rng()
    >>> img += 0.1 * img.std() * rng.standard_normal(img.shape)
    >>> deconvolved_img = restoration.unsupervised_wiener(img, psf)

    Notes
    -----
    The estimated image is design as the posterior mean of a
    probability law (from a Bayesian analysis). The mean is defined as
    a sum over all the possible images weighted by their respective
    probability. Given the size of the problem, the exact sum is not
    tractable. This algorithm use of MCMC to draw image under the
    posterior law. The practical idea is to only draw highly probable
    images since they have the biggest contribution to the mean. At the
    opposite, the less probable images are drawn less often since
    their contribution is low. Finally, the empirical mean of these
    samples give us an estimation of the mean, and an exact
    computation with an infinite sample set.

    References
    ----------
    .. [1] François Orieux, Jean-François Giovannelli, and Thomas
           Rodet, "Bayesian estimation of regularization and point
           spread function parameters for Wiener-Hunt deconvolution",
           J. Opt. Soc. Am. A 27, 1593-1607 (2010)

           https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-27-7-1593

           https://hal.archives-ouvertes.fr/hal-00674508
    """
    params = {'threshold': 0.0001, 'max_num_iter': 200, 'min_num_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})
    if reg is None:
        reg, _ = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.real.astype(float_type, copy=False)
    reg = reg.real.astype(float_type, copy=False)
    if psf.shape != reg.shape:
        trans_fct = uft.ir2tf(psf, image.shape, is_real=is_real)
    else:
        trans_fct = psf
    x_postmean = np.zeros(trans_fct.shape, dtype=float_type)
    prev_x_postmean = np.zeros(trans_fct.shape, dtype=float_type)
    delta = np.nan
    gn_chain, gx_chain = ([1], [1])
    areg2 = np.abs(reg) ** 2
    atf2 = np.abs(trans_fct) ** 2
    if is_real:
        data_spectrum = uft.urfft2(image)
    else:
        data_spectrum = uft.ufft2(image)
    rng = np.random.default_rng(rng)
    for iteration in range(params['max_num_iter']):
        precision = gn_chain[-1] * atf2 + gx_chain[-1] * areg2
        _rand1 = rng.standard_normal(data_spectrum.shape)
        _rand1 = _rand1.astype(float_type, copy=False)
        _rand2 = rng.standard_normal(data_spectrum.shape)
        _rand2 = _rand2.astype(float_type, copy=False)
        excursion = np.sqrt(0.5 / precision) * (_rand1 + 1j * _rand2)
        wiener_filter = gn_chain[-1] * np.conj(trans_fct) / precision
        x_sample = wiener_filter * data_spectrum + excursion
        if params['callback']:
            params['callback'](x_sample)
        gn_chain.append(rng.gamma(image.size / 2, 2 / uft.image_quad_norm(data_spectrum - x_sample * trans_fct)))
        gx_chain.append(rng.gamma((image.size - 1) / 2, 2 / uft.image_quad_norm(x_sample * reg)))
        if iteration > params['burnin']:
            x_postmean = prev_x_postmean + x_sample
        if iteration > params['burnin'] + 1:
            current = x_postmean / (iteration - params['burnin'])
            previous = prev_x_postmean / (iteration - params['burnin'] - 1)
            delta = np.sum(np.abs(current - previous)) / np.sum(np.abs(x_postmean)) / (iteration - params['burnin'])
        prev_x_postmean = x_postmean
        if iteration > params['min_num_iter'] and delta < params['threshold']:
            break
    x_postmean = x_postmean / (iteration - params['burnin'])
    if is_real:
        x_postmean = uft.uirfft2(x_postmean, shape=image.shape)
    else:
        x_postmean = uft.uifft2(x_postmean)
    if clip:
        x_postmean[x_postmean > 1] = 1
        x_postmean[x_postmean < -1] = -1
    return (x_postmean, {'noise': gn_chain, 'prior': gx_chain})