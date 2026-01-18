from dissononce.processing.impl.symmetricstate import SymmetricState
WA does not mix_hash if cipherstate does not yet have a key, therefore this custom WASymmetricState has to
    be used instead of default one.