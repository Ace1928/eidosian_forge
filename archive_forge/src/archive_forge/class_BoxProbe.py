import numpy as np
class BoxProbe:
    """Box shaped Buttinger probe.
    
    Kramers-kroning: real = H(imag); imag = -H(real)
    """

    def __init__(self, eta, a, b, energies, S, T=0.3):
        from Transport.Hilbert import hilbert
        se = np.empty(len(energies), complex)
        se.imag = 0.5 * (np.tanh(0.5 * (energies - a) / T) - np.tanh(0.5 * (energies - b) / T))
        se.real = hilbert(se.imag)
        se.imag -= 1
        self.selfenergy_e = eta * se
        self.energies = energies
        self.S = S

    def retarded(self, energy):
        return self.selfenergy_e[self.energies.searchsorted(energy)] * self.S