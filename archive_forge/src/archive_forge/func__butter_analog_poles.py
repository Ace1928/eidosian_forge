def _butter_analog_poles(n):
    """
    Poles of an analog Butterworth lowpass filter.

    This is the same calculation as scipy.signal.buttap(n) or
    scipy.signal.butter(n, 1, analog=True, output='zpk'), but mpmath is used,
    and only the poles are returned.
    """
    poles = [-mpmath.exp(1j * mpmath.pi * k / (2 * n)) for k in range(-n + 1, n, 2)]
    return poles