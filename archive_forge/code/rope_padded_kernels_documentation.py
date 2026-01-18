import triton  # type: ignore
import triton.language as tl  # type: ignore

    Each letter in this diagram is a whole row of length dim.

     INPUT      xq        xk       xv

        head_dim ─►

      batch   qqqqqq      kk       vv
        │     qqqqqq      kk       vv
        ▼     qqqqqq      kk       vv

    head_idx:  (goes across all heads of all 3 inputs)
              ▲     ▲     ▲ ▲      ▲ ▲
              │     │     │ │      │ │
                          │        │
              0  k_start  │v_start │n_total_heads
                          │        │
                          │        │
                      k_start    v_start

    Output is to out_q (same shape as xq), an xk-shaped part
    of cache_k and an xv-shaped part of cache_v
    