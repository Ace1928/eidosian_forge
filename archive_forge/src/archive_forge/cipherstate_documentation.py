
        DecryptWithAd(ad, ciphertext):
        If k is non-empty returns DECRYPT(k, n++, ad, ciphertext). Otherwise returns ciphertext.
        If an authentication failure occurs in DECRYPT() then n is not incremented
        and an error is signaled to the caller.

        :param ad:
        :type ad: bytes
        :param ciphertext:
        :type ciphertext: bytes
        :return: bytes
        :rtype:
        