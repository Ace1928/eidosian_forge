from Bio import motifs
def _read_lpm(handle, num_occurrences):
    """Read letter probability matrix (PRIVATE)."""
    counts = [[], [], [], []]
    for line in handle:
        freqs = line.split()
        if len(freqs) != 4:
            break
        counts[0].append(round(float(freqs[0]) * num_occurrences))
        counts[1].append(round(float(freqs[1]) * num_occurrences))
        counts[2].append(round(float(freqs[2]) * num_occurrences))
        counts[3].append(round(float(freqs[3]) * num_occurrences))
    c = {}
    c['A'] = counts[0]
    c['C'] = counts[1]
    c['G'] = counts[2]
    c['T'] = counts[3]
    return c