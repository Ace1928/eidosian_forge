Titans: Learning to Memorize at Test Time
Ali Behrouz†
, Peilin Zhong†
, and Vahab Mirrokni†
†
Google Research
{alibehrouz, peilinz, mirrokni}@google.com
Abstract
Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and
attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows
attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling
of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new
neural long-term memory module that learns to memorize historical context and helps an attention to attend to the
current context while utilizing long past information. We show that this neural memory has the advantage of a fast
parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its
limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its
ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce
a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate
memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics,
and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models.
They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks
compared to baselines.
1 Introduction “The true art of memory is the art of attention!"
— Samuel Johnson, 1787
Transformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-of-
the-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan
et al. 2020). The primary building blocks of Transformers–attention modules—function as associative memory
blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise
similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer
is exclusively conditioned on the direct dependencies of tokens in the current context window. This accurate modeling of
dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex
real-world tasks (e.g., language modeling (N. F. Liu et al. 2024), video understanding (C. -Y. Wu et al. 2019), long-term time
series forecasting (H. Zhou et al. 2021)), the context window can become extremely large, making the applicability of
Transformers challenging in these downstream tasks.
To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transform-
ers (Kacham, Mirrokni, and P. Zhong 2024; Katharopoulos et al. 2020; S. Yang, B. Wang, Shen, et al. 2024), where softmax is
replaced by a kernel function in the attention (see §2.1 for details), resulting in a significant drop in memory consumption.
Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance
compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed
into a matrix-valued states (Katharopoulos et al. 2020). This, however, brings a contradictory fact about linear recurrent (or
linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs.
quadratic complexity), whose advantages is appeared for very long context; On the other hand, a very long context cannot
be properly compressed in a small vector-valued or matrix-valued states (S. Wang 2024).
1
arXiv:2501.00663v1 [cs.LG] 31 Dec 2024
Furthermore, beyond efficiency, most existing architectures–ranging from Hopfield Networks (Hopfield 1982) to LSTMs (Jür-
gen Schmidhuber and Hochreiter 1997) and Transformers (Vaswani et al. 2017)–face challenges when dealing with general-
ization, length extrapolation, and/or reasoning (Anil et al. 2022; Qin, Y. Zhong, and Deng 2024), all of which are inseparable
parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each of which
are missing: (1) a crucial component for learning process—such as short-term memory, long-term memory, meta-memory,
attending to current context, etc. (Cowan 2008); (2) how these components are interconnected systems that can operate
independently; and/or (3) the ability to actively learn from data and memorize the abstraction of past history. We argue
that in an effective learning paradigm, similar to human brain, there are distinct yet interconnected modules, each of which
is responsible for a component crucial to the learning process.
Memory Perspective
Memory is a fundamental mental process and is an inseparable component of human learning (Terry 2017). Without
a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped
behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g.,
Hopfield Networks (Hopfield 1982), LSTMs (Jürgen Schmidhuber and Hochreiter 1997), and Transformers (Vaswani et al.
2017).
Taking inspiration from the common definitions of memory and learning in neuropsychology literature (Okano, Hirano,
and Balaban 2000), most existing architectures consider memory as a neural update caused by an input, and define learning
as a process for acquiring effective and useful memory, given an objective. In this perspective, Recurrent Neural Networks
(RNNs) (Williams and Zipser 1989) can be defined as models with a vector-valued memory module M (also called hidden
state) with two main steps: Given a new input 𝑥𝑡 at time 𝑡, the model (1) updates the memory using a function 𝑓 (M𝑡 −1, 𝑥𝑡 )
(with compression); and (2) retrieves the corresponding memory of input using a function 𝑔(M𝑡 , 𝑥𝑡 ) (see §2.1 for details).
Similarly, Transformers can be seen as architectures with a growing memory and two similar steps. That is, the pair of key
and value matrices acts as the model’s memory, and the model: (1) updates the memory by appending the key and value to
the memory (without compression), and (2) retrieves query vectors’ corresponding memory by finding the similarity of
query and key vectors, which is then used to weight the value vectors for the output.
This perspective, can help us better understand existing paradigms, their critical differences, and design more effective
architectures. For example, the main difference between Transformers (Vaswani et al. 2017) and linear Transform-
ers (Katharopoulos et al. 2020) is the memory structure as well as the memory updating step, in which linear Transformers
compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within
the context length) without any compression. While both linear Transformers and linear RNNs (including state space
models) compress the information in memory update step, the critical difference lies in the structure of the memory,
where linear RNNs (vs. linear Transformers) use a vector-valued memory (vs. matrix-valued memory). Therefore, this
perspective motivates us to ask: (Q1) What constitute a good structure for the memory? (Q2) What is a proper memory
update mechanism? and (Q3) What is a good memory retrieval process?
Revisiting our understanding of human memory, it is neither a unitary process nor it serves a single function (Cowan
2008). In fact, memory is a confederation of systems–e.g., short-term, working, and long-term memory–each serving a
different function with different neural structures, and each capable of operating independently (Willingham 1997). This
fact motivates us to ask: (Q4) How to design an efficient architecture that incorporates different interconnected memory
modules. Finally, storing a memory is a neural process that requires to encode and store the abstraction of the past. It can
be over-simplification to assume a single vector or a matrix, whose parameters are encoding the data in a linear manner,
are enough for storing long-term history. (Q5) Is a deep memory module needed to effectively store/remember long
past?
Contributions and Roadmap
In this paper, we aim to answer the above five questions by designing a long-term neural memory module, that can
efficiently and effectively learn to memorize at test time. Building upon its design, we discuss how it can be incorporated
into an architecture.
Neural Memory (§3). We present a (deep) neural long-term memory that (as a meta in-context model) learns how to
memorize/store the data into its parameters at test time. Inspired by human long-term memory system (Mandler 2014),
2
we design this memory module so an event that violates the expectations (being surprising) is more memorable. To this
end, we measure the surprise of an input with the gradient of the neural network with respect to the input in associative
memory loss (see §3.1 for details). To better handle the limited memory, we present a decaying mechanism that consider the
proportion of memory size and the amount of data surprise, resulting in better memory management. We show that this
decay mechanism is in fact the generalization of forgetting mechanism in modern recurrent models (Dao and Gu 2024; Gu
and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024). Interestingly, we find that this mechanism is equivalent to optimizing
a meta neural network with mini-batch gradient descent, momentum, and weight decay. Building upon tensorizing
mini-batch gradient descent to use more matmul operations (Yu Sun et al. 2024), we present a fast and parallelizable
algorithm to train our deep neural long-term memory.
Titans Architectures (§4). After designing the long-term neural memory, an important remaining question is how to
effectively and efficiently incorporate memory into a deep learning architecture. We present Titans, a family of deep models
that consists of three hyper-heads: (1) Core: this module consists of the short-term memory, and is responsible for the main
flow of processing the data (we use attention with limited window size); (2) Long-term Memory: this branch is our neural
long-term memory module that is responsible to store/remember long past; (3) Persistent Memory: this is a set of learnable
but date-independent parameters that encodes the knowledge about a task. Finally, as a proof of concept, we present three
variants of Titans, in which we incorporate memory as: (i) a context, (ii) a layer, and (iii) a gated branch.
Experimental Results (§5). We perform experimental evaluations on language modeling, commonsense reasoning, recall-
intensive, needle in haystack, time series forecasting, and DNA modeling tasks. We observe that our Titan architecture
outperforms all modern recurrent models as well as their hybrid variants (combining with sliding-window attention) across
a comprehensive set of benchmarks. Furthermore, Titans outperforms Transformers with the same context window, and
show competitive performance with Transformers that use the entire context. This results are achieved while, contrary to
Transformers, Titans scale to larger than 2M context window size.
2 Preliminaries
In this section, we discuss the notation and some background concepts that we use though the paper. We let
𝑥 ∈ R𝑁 ×𝑑in be the input, M be a neural network (neural memory module), Q, K, V be the query, key and value
of the attention mechanism, and M be the attention mask. When segmenting the sequence, we use S(𝑖 ) to refer to
the 𝑖-th segment. Through the paper, we abuse the notation and use subscripts to refer to a specific element of a matrix,
vector, or segments. For example, we let S(𝑖 )
𝑗 be the 𝑗-th token in the 𝑖-th segment. The only exception is subscripts with 𝑡,
which we reserved to index recurrence over time, or the state of a neural network at time 𝑡. Given a neural network N and
a data sample 𝑥, we use N (𝑥) (resp. N∗ (𝑥)) to refer to the forward pass with (resp. without) weight adjustment. Also, we
abuse the notation and use N (𝑘 ) to refer to the 𝑘-th layer of the neural network. In the following, we first, discuss the
backgrounds for attention and its efficient variants followed by a review of modern linear RNNs. Finally, we discuss a
memory perspective of these architectures that motivates us to design Titans.
2.1 Backgrounds
Attention. Transformers (Vaswani et al. 2017) as the de facto backbone for many deep learning models are based on
attention mechanism. Given input 𝑥 ∈ R𝑁 ×𝑑in , causal attention computes output y ∈ R𝑁 ×𝑑in based on softmax over input
dependent key, value, and query matrices:
Q = 𝑥WQ, K = 𝑥WK, V = 𝑥WV, (1)
y𝑖 =
𝑖∑︁
𝑗=1
exp

Q⊤
𝑖 K𝑗 /√𝑑in

V𝑗
Í𝑖
ℓ=1 exp

Q⊤
𝑖 Kℓ /√𝑑in
 , (2)
where WQ, WK, and WV ∈ R𝑑in ×𝑑in are learnable parameters. Despite the power and effectiveness in recall, transformers
need at least 𝑁 × 𝑑 operators to calculate the output, resulting in larger memory consumption and lower-throughput for
longer sequences.
Efficient Attentions. To improve the memory consumption and throughput of softmax attention for longer sequences,
various studies focused on I/O aware implementations of attention (Dao 2024; Dao, D. Fu, et al. 2022), designing more
3
efficient attention mechanisms by sparsifying the attention matrix (B. Chen et al. 2021; Choromanski et al. 2021; Dai et al.
2019), approximating the softmax (Arora et al. 2024), or developing kernel-based (linear) attentions (Aksenov et al. 2024;
Kacham, Mirrokni, and P. Zhong 2024; Schlag, Irie, and Jürgen Schmidhuber 2021; S. Yang, B. Wang, Shen, et al. 2024). In
this part, we focus on the later, i.e., linear attentions, where the softmax in standard attention is replaced with an alternative
kernel function 𝜙 (., .), such that 𝜙 (𝑥, 𝑦) = 𝜙 (𝑥)𝜙 (𝑦). Accordingly, the attention can be written as:
y𝑖 =
𝑖∑︁
𝑗=1
𝜙 (𝑄⊤
𝑖 𝐾𝑗 )
Í𝑖
ℓ=1 𝜙 (𝑄⊤
𝑖 𝐾ℓ ) 𝑉𝑗 =
𝑖∑︁
𝑗=1
𝜙 (𝑄𝑖 )⊤𝜙 (𝐾𝑗 )
Í𝑖
ℓ=1 𝜙 (𝑄𝑖 )⊤𝜙 (𝐾ℓ ) 𝑉𝑗 = 𝜙 (𝑄𝑖 )⊤ Í𝑖𝑗=1 𝜙 (𝐾𝑗 )𝑉𝑗
𝜙 (𝑄𝑖 )⊤ Í𝑖
ℓ=1 𝜙 (𝐾ℓ ) , (3)
resulting in a higher-throughput as terms Í𝑖𝑗=1 𝜙 (𝐾𝑗 ) and Í𝑖
ℓ=1 𝜙 (𝐾ℓ ) are re-using in each step. When choosing the kernel
as identity matrix (Yutao Sun et al. 2023), the above formulation can also be written in a recurrent format:
M𝑡 = M𝑡 −1 + 𝐾⊤
𝑡 𝑉𝑡 , (4)
y𝑡 = 𝑄𝑡 M𝑡 , (5)
which allows efficient inference for linear attentions.
Modern Linear Models and Their Memory Perspective. As discussed earlier, one can define learning as a process for
acquiring effective and useful memory. Building upon this, one can see the hidden state of Recurrent Neural Networks
(RNNs) as a memory unit, which the model aims to compress the information into. Accordingly, in a general form of
recurrent neural network, the hidden state can be treated as a memory unit and the recurrence process can be split into the
read and write operations in the memory unit. That is, we let 𝑥 ∈ R𝑁 ×𝑑in be the input, M ∈ R𝑑 is the memory unit, and
y ∈ R𝑑in is the output, then the general form of the recurrent neural network is defined as:
M𝑡 = 𝑓 (M𝑡 −1, 𝑥𝑡 ), Write Operation (6)
y𝑡 = 𝑔(M𝑡 , 𝑥𝑡 ), Read Operation (7)
where 𝑓 (., .) is the read and 𝑔(., .) is the write corresponding functions. Note that here the subscript of M𝑡 shows the state
of the memory at time 𝑡.
In this perspective, the recurrence formula of linear Transformers (see Equation 4) is equivalent to additively compress
and write keys and values, (𝐾𝑡 , 𝑉𝑡 ), into a matrix-valued memory unit M𝑡 . Therefore, when dealing with long context
data, this additive nature of the process results in memory overflow, significantly damaging the performance of the model.
To address this, studies have focused on two promising directions: (1) Adding forget mechanism: several studies have
presented adaptive (data-dependent) forgetting gate mechanisms for linear models, where it can erase the memory when it
is needed. As examples of such models, we refer to GLA (S. Yang, B. Wang, Shen, et al. 2024), LRU (Orvieto et al. 2023),
Griffin (De et al. 2024), xLSTM (Beck et al. 2024), and Mamba2 (Dao and Gu 2024), which the later is also connected to the
discretized version of traditional state space models (Gu and Dao 2024).(2) Improving the write operation: To overcome the
additive nature of memory write operation in traditional recurrent models, Widrow and Hoff (1988) presented Delta Rule,
in which before adding a memory (i.e., a pair of key and value), the model first removes its past value. To enhance the
parallelizable training and scaling, S. Yang, B. Wang, Yu Zhang, et al. (2024) present a fast paralellizable algorithm. Finally,
very recently, S. Yang, Kautz, and Hatamizadeh (2024) improved the DeltaNets by adding a forget gate.
Memory Modules. Memory has always been one of the core parts of the neural network designs (Graves, Wayne,
and Danihelka 2014; JH Schmidhuber 1992; Jürgen Schmidhuber and Hochreiter 1997; J. Zhang et al. 2024). The idea of
seeing linear layers as the key-value (associative) memory system backs to fast weight programs, in which dynamic fast
programs are incorporated into recurrent neural networks to serve as writable memory (JH Schmidhuber 1992). The two
learning rules of Hebbian (Hebb 2005) and delta (Prados and Kak 1989) are the most popular learning rules for fast weight
programs, which have been extensively explored in various studies (Irie, Schlag, et al. 2021; Munkhdalai, Sordoni, et al.
2019; Munkhdalai and H. Yu 2017; Schlag, Irie, and Jürgen Schmidhuber 2021; JH Schmidhuber 1992; S. Yang, Kautz, and
Hatamizadeh 2024; S. Yang, B. Wang, Yu Zhang, et al. 2024). All these models, however, are based on momentary surprise,
missing the token flow in the sequences (see Section 3.1), and most of them lacks a forgetting gate, resulting in a poor
memory management.
We further discuss the connection of our architectures with recent models in Appendix C. Additional related work are
discussed in Appendix A.
4
3 Learning to Memorize at Test Time
To overcome the lack of long-term memory and to enable the model to learn, forget, and retrieve information, in
this section, we present a neural long-term memory module, which is a meta models that learns to memorize at
test time. In Section 3.1, we first discuss the motivation and the design of the neural memory. In Section 3.2, we
discuss how our architecture design can benefit from a fast and parallelizable training. Finally, in Section 3.3, we augment
our architecture using persistent memory module, in which we use learnable but data-independent parameters to learn
meta information about the task.
3.1 Long-term Memory
To design a neural long-term memory module, we need a model that can encode the abstraction of the past history into its
parameters. An example of this can be LLMs that are shown to be memorizing their training data (Leybzon and Kervadec
2024; Schwarzschild et al. 2024; Staab et al. 2024). Therefore, a simple idea is to train a neural network and expect it to
memorize its training data. Memorization, however, has almost always been known as an undesirable phenomena in
neural networks as it limits the model generalization (Bayat et al. 2024), causes privacy concerns (Staab et al. 2024), and
so results in poor performance at test time. Moreover, the memorization of the training data might not be helpful at test
time, in which the data might be out-of-distribution. We argue that, we need an online meta-model that learns how to
memorize/forget the data at test time. In this setup, the model is learning a function that is capable of memorization, but it
is not overfitting to the training data, resulting in a better generalization at test time.
Learning Process and Surprise Metric. The key idea to train a long-term memory is to treat its training as an online
learning problem, in which we aim to compress the past information 𝑥1, . . . , 𝑥𝑡 −1 into the parameters of our long-term
neural memory module M𝑡 . As discussed earlier, an event that violates the expectations (i.e., is surprising) is more
memorable for humans (Mandler 2014). Inspired by this, a simple definition of surprise for a model can be its gradient with
respect to the input. The larger the gradient is, the more different the input data is from the past data. Accordingly, using
this surprise score, we can update the memory as:
M𝑡 = M𝑡 −1 − 𝜃𝑡 ∇ℓ (M𝑡 −1; 𝑥𝑡 )
| {z }
Surprise
. (8)
This surprise metric, however, can result in missing important information that comes after a big surprising moment.
That is, the gradient can become extremely small after several surprising steps, leading to stocking in a flat area (i.e., local
minima), and missing information about some parts of the sequence. From the human memory perspective, an event might
not consistently surprise us through a long-period of time although it is memorable. The reason is that the initial moment
is surprising enough to get our attention through a long time frame, leading to memorizing the entire time frame. To
improve the above surprise metric (Equation 8), we break the surprise metric into (1) past surprise, which measures the
surprise amount of a very recent past; and (2) momentary surprise, which measures the surprise of incoming data:
M𝑡 = M𝑡 −1 + 𝑆𝑡 , (9)
𝑆𝑡 = 𝜂𝑡 𝑆𝑡 −1
|{z}
Past Surprise
− 𝜃𝑡 ∇ℓ (𝑀𝑡 −1; 𝑥𝑡 )
| {z }
Momentary Surprise
. (10)
Interestingly, this formulation is similar to gradient descent with momentum, where 𝑆𝑡 is the momentum element. Therefore,
the momentum here act as a memory of surprise across time (sequence length). In this formulation, the term 𝜂𝑡 is a
data-dependent surprise decay (a function of 𝑥𝑡 ), controlling how surprise decays over time, and the term 𝜃𝑡 is controlling
how much of momentary surprise should be incorporated into the final surprise metric in a data-dependent manner. This
data-dependency is particularly important in this design: While surprise of previous tokens might be needed to affect
the surprise of the next token, it is mostly valid if all tokens are relevant and are in the same context. Accordingly, a
data-dependent 𝜂 can control if memory needs to: (1) ignore the last surprise by setting 𝜂𝑡 → 0 (possibly due to the change
of context), or (2) fully incorporate the last surprise by setting 𝜂𝑡 → 1 (possibly as the token is highly relevant to its recent
past tokens).
Objective. Our above surprise metric is based on a loss function ℓ (.; .), which is the objective that our memory is learning
to act as it at test time. That is, our memory module is a meta model that learns a function based on the loss function ℓ (.; .).
5
In this work, we focus on associative memory, in which we aim to store the past data as the pairs of keys and values. Given
𝑥𝑡 , similar to Transformers (Vaswani et al. 2017), we use two linear layers to project 𝑥𝑡 into a key and value:
k𝑡 = 𝑥𝑡𝑊𝐾 , v𝑡 = 𝑥𝑡𝑊𝑉 , (11)
where 𝑊𝐾 and 𝑊𝑉 ∈ R𝑑in ×𝑑in . Next, we expect our memory module to learn the associations between keys and values. To
this end, we define the loss as follows:
ℓ (M𝑡 −1; 𝑥𝑡 ) = ∥M𝑡 −1 (k𝑡 ) − v𝑡 ∥2
2 (12)
By optimizing the above loss function in the inner-loop of our meta model (memory), the model learns how to memorize
the mapping between keys and values at test time. Note that, similar to meta-learning models (Nichol 2018; Zintgraf et al.
2019), training of the memory is in the inner-loop, and so parameters 𝑊𝐾 and 𝑊𝑉 are hyperparameters in the above loss
function. Accordingly, in the inner loop, we optimize M’s weights, while in the outer-loop, we optimize other parameters
of the entire architecture.
Forgetting Mechanism. When dealing with very large sequences (e.g., millions of tokens), it is crucial to manage which
past information should be forgotten–even with a deep or a very large matrix-valued memory. To this end, we use an
adaptive forgetting mechanism that allows the memory to forget the information that is not needed anymore, resulting in
better managing the memory’s limited capacity. That is, given the next token 𝑥𝑡 , we modify the update rule as:
M𝑡 = (1 − 𝛼𝑡 )M𝑡 −1 + 𝑆𝑡 , (13)
𝑆𝑡 = 𝜂𝑡 𝑆𝑡 −1 − 𝜃𝑡 ∇ℓ (𝑀𝑡 −1; 𝑥𝑡 ), (14)
where 𝛼𝑡 ∈ [0, 1] is the gating mechanism that flexibly controls the memory; i.e., decides how much information should be
forgotten. For example, it can update the memory without affecting the past abstraction by letting 𝛼𝑡 → 0, and can clear
the entire memory by letting 𝛼𝑡 → 1. Later in this section, we show that this weight decay mechanism is closely related to
the gating mechanism in modern RNNs (Dao and Gu 2024; Orvieto et al. 2023).
Memory Architecture. In this paper, we focus on simple MLPs with 𝐿M ≥ 1 layers as the architecture of our long-term
memory. The main reason behind this choice is that we want to focus on better motivating the design of the long-term
memory and ways that it can be incorporated into an architecture. However, our formulation and architectural design
opens a new research direction to design neural architectures that are more effective and efficient in memorization of data.
Recently, there has been a promising line of work to design such architectures (Berges et al. 2024; Cetin et al. 2024; J. Zhang
et al. 2024), which incorporating them into our framework (i.e., replacing simple MLPs with such architectures) can be an
interesting future work.
When using vector-valued or matrix-valued memory (De et al. 2024; Orvieto et al. 2023; S. Yang, B. Wang, Shen, et
al. 2024), the memory module is compressing the past data and fit it into a line. That is, from the meta learning or
online learning perspective (Yu Sun et al. 2024), using a matrix-valued memory M = 𝑊 ∈ R𝑑in ×𝑑in is equivalent to
optimize ℓ (𝑊𝑡 −1; 𝑥𝑡 ) = ∥𝑊𝑡 −1k𝑡 − v𝑡 ∥2
2, which is an online linear regression objective and so the optimal solution assumes
the underlying dependency of historical data is linear. On the other hand, we argue that deep memory modules (i.e.,
𝐿M ≥ 2) . Aligning with the theoretical results that MLPs with at least two layers are strictly more expressive than linear
models (Hornik, Stinchcombe, and White 1989), in Section 5.5, we show that deep memory modules are more effective in
practice.
Retrieving a Memory. In the above, we discuss how one can design and train a long-term memory module that learns to
memorize at test time. A key remaining question is: How one can retrieve information from the memory? We simply use the
forward pass without weight update (i.e., inference) to retrieve a memory correspond to a query. Formally, given an input
𝑥𝑡 , we use a linear layer 𝑊𝑄 to project the input, i.e., q𝑡 = 𝑥𝑡𝑊𝑄 and retrieve the corresponding (or useful) information
from the memory 𝑦𝑡 by:
𝑦𝑡 = M∗ (q𝑡 ). (15)
6
Figure 1: The illustration of how the training of neural memory can be done in parallel and using matmuls.
3.2 How to Parallelize the Long-term Memory Training
As discussed above, the design of our long-term memory module is equivalent to training a meta model by optimizing
associative memory loss function ℓ (M𝑡 −1; 𝑥𝑡 ) = ∥M𝑡 −1 (k𝑡 ) − v𝑡 ∥2
2 using gradient descent with momentum and weight
decay. Therefore, in theory, the training of long-term memory module requires O (𝑁 ) FLOPs, where 𝑁 is the sequence
length. However, in practice, we need to parallelize the training process and to fully take advantage of hardware accelerators
(e.g., TPUs, GPUs), we need to tensorize the process and use more matmuls.
Next, we show that calculating the weights in the inner loop with mini-batch gradient descent, data-dependent learning
rate, and weight decay can be reformulated so that it uses only matmuls and sum. We build upon the work of Yu Sun et al.
(2024) that shows forward pass of a model optimizing with the mini-batch gradient descent (with constant learning rate)
can be calculated using matmuls. We can split the sequence into chunks of size 𝑏 ≥ 1, and write the mini-batch gradient
descent as:
M𝑡 = (1 − 𝛼𝑡 )M𝑡 −1 − 𝜃𝑡 ∇ℓ (M𝑡 −1; 𝑥𝑡 ) = 𝛽𝑡 M0 −
𝑡∑︁
𝑖=1
𝜃𝑖
𝛽𝑡
𝛽𝑖
∇ℓ (M𝑡 ′ ; 𝑥𝑖 ), (16)
where 𝑡′ = 𝑡 − mod(𝑡, 𝑏), and 𝛽𝑖 = Î𝑖𝑗=1 (1 − 𝛼 𝑗 ). For the sake of simplicity, we focus on the first chunk, i.e., 𝑡 = 𝑏 and so
𝑡′ = 0. Also, we explain the process for the case that M𝑡 = 𝑊𝑡 is linear. The process for MLPs with 𝑁𝑝 ≥ 2 is similar. Using
our loss function, we have:
∇ℓ (𝑊0; 𝑥𝑡 ) = (𝑊0𝑥𝑡 − 𝑥𝑡 )𝑥⊤
𝑡 ⇒
𝑏∑︁
𝑖=1
𝜃𝑖
𝛽𝑏
𝛽𝑖
∇ℓ (𝑊0; 𝑥𝑖 ) = Θ𝑏 B𝑏 (𝑊0𝑋 − 𝑋 )𝑋 ⊤, (17)
where Θ𝑏 = diag  𝜃1 𝜃2 . . . 𝜃𝑏
  and B𝑏 is defined analogously on 𝛽𝑏
𝛽𝑖 s. Note that, we do not need to store all Θ𝑘𝑏 and
B𝑘𝑏 for 𝑘 = 1, . . . , 𝑁 /𝑏, instead, we store these matrices for each chunk, resulting in using less memory. Next, we extend
this representation so we can also incorporate the momentum term. In a chunk wise gradient descent with momentum, if
we look at the momentum term, we have:
𝑆𝑡 = 𝜂𝑡 𝑆𝑡 −1 − 𝜃𝑡 𝑢𝑡 , (18)
where 𝑢𝑡 = ∇ℓ (𝑀𝑡 ′ ; 𝑥𝑡 ). Note that, we can compute all 𝑢𝑡 at the same time, and so Equation 18 is a linear recurrence
with 𝑢𝑡 as an input, 𝑆𝑡 as the hidden state, and 𝜂𝑡 as input-dependent transition value. Accordingly, we can use parallel
associative scan (J. T. Smith, Warrington, and Linderman 2023) to calculate 𝑆𝑡 s in this chunk.
Parameters as the Function of Chunks. Instead of making parameters like 𝛼𝑡 , 𝜃𝑡 , and 𝜂𝑡 input-dependent (i.e., a function
of token 𝑥𝑡 ), we can make them functions of their chunk. Despite losing expressive power, this formulation can help to
make the training even faster. In this case, we are using the same value for each of 𝛼, 𝜃 , and 𝜂 in each chunk. Accordingly,
in Equation 17, we can store Θ using a single scaler. Similarly we can make Equation 18 faster. That is, when 𝜂 and 𝜃 are
learnable but time-invariant inside each chunk, this equation becomes a linear time-invariant system (LTI), which can be
computed by a global convolution (Gu, Goel, and Re 2022). In our experiments, we make these parameters as the functions
of tokens. However, such simplifications (i.e., as the function of chunks) can be the interest of future work to training
larger models in more efficient manner.
7
Figure 2: Memory as a Context (MAC) Architecture. This architecture includes three branches of (1) core, (2) contextual
(long-term) memory, and (3) persistent memory. The core branch concatenates the corresponding long-term and persistent
memories with the input sequence. Next, attention performs on the sequence and decides what part of the information
should store in the long-term memory. At the test time, parameters corresponds to contextual memory are still learning,
parameters corresponds to the core branch are responsible for in-context learning, and parameters of persistent memory
are responsible to store the knowledge about tasks and so are fixed.
3.3 Persistent Memory
Our long-term memory can also be seen as a contextual memory, meaning that the output is fully depend on the context.
Therefore, in addition to our long-term memory, we also use a set of learnable but input-independent parameters to act as
task-related memory. This type of memory has been referred to as persistent or meta-memory in the literature (X. Dong
et al. 2024; Sukhbaatar, Grave, et al. 2019). Given 𝑁𝑝 ≥ 1, we use learnable parameters 𝑃 = 𝑝1 𝑝2 . . . 𝑝𝑁𝑝
 and
append it to the start of our sequence: i.e., given a context window size of 𝑁 , we modify the input as:
𝑥new = 𝑝1 𝑝2 . . . 𝑝𝑁𝑝
 || 𝑥, (19)
where || is concatenation. Next, we discuss the motivation of persistent memory from three perspective:
Memory Perspective. As discussed earlier, our neural long-term memory is a contextual memory, in which all parameters
are input-dependent. An effective memory system, however, also needs input-independent parameters to store the
abstraction of the task knowledge. That is, mastering a task requires the memorization of the knowledge that how the task
can be done, and these parameters are responsible for storing such knowledge.
Feedforward Network Perspective. In the Transformer architectures, there are fully connected layers after the attention
module, which are shown to be similar to attention weights but with data-independent parameters. That is, Sukhbaatar,
Grave, et al. (2019) showed that replacing the ReLU in fully connected layers with Softmax can results in an attention-like
weights, in which weights are data-independent:
𝐹 𝐹 𝑁 (𝑥) = 𝑊𝑉 Softmax (𝑊𝐾 𝑥) . (20)
In fact, 𝑊𝐾 and 𝑊𝑉 are acting similar to 𝐾 and 𝑉 matrices in attention module when they are input-independent. The
persistent memory weights are expected to have the same functionality, meaning that using them in the first part of the
sequence leads to having input-independent attention weights (Sukhbaatar, Grave, et al. 2019).
Technical Perspective. Attention with causal mask has implicit bias toward initial tokens in the sequence, and so attention
weights are almost always highly active for initial tokens, resulting in performance damage. From the technical perspective,
these learnable parameters at the start of the sequence can mitigate such effect by redistributing the attention weights
more effectively (Han et al. 2024; Xiao et al. 2024).
8
(a) Memory as a Context (MAC). We segment the sequence
and use full causal attention in each window. Again, the first
𝑁𝑝 tokens are persistent memory and the next 𝑁𝑙 are long-term
memory tokens
(b) Memory as Gating (MAG). We use sliding window attention
(SWA) as a short-term memory and our neural memory module
as a long-term memory, combining by a gating.
Figure 3: Attention masks for different variants of Titans.
4 How to Incorporate Memory?
An important question that remained unanswered is: How one can effectively and efficiently incorporate the
designed neural memory into a deep learning architecture? As discussed earlier, from a memory perspective,
the pair of K and V matrices in transformers can be interpreted as an associative memory block. Due to their
accurate modeling of dependencies and so their limited context window, we interpret them as short-term memory modules,
attending to the current context window size. On the other hand, our neural memory with the ability to continuously
learn from data and store it in its weights can play the role of a a long-term memory. In this section, we aim to answer
the above question by proposing three different variants of Titans. Later in our experiments, we show that each of these
variants has its own advantages/disadvantages and also can show a trade-off between the efficiency and effectiveness in
very long-contexts.
4.1 Memory as a Context
In the first architecture design (see Figure 2), we treat the memory as a context to the current information. That is, given
a long sequence 𝑥 ∈ R𝑁 ×𝑑in , we first chunk the sequence into fixed-size segments S(𝑖 ) for 𝑖 = 1, . . . , 𝑁 /𝐶. Given the
incoming segment S(𝑡 ) , we consider it as the current context and its past segment as the historical information. Therefore,
let M𝑡 −1 be the state of long-term memory before segment S(𝑡 ) , we use the input context as the query to the memory
M𝑡 −1 to retrieve the corresponding information from the long-term memory. That is, we retrieve the past information that
corresponds to S(𝑡 ) as:
ℎ𝑡 = M∗
𝑡 −1 (q𝑡 ), (21)
where q𝑡 = S(𝑡 )𝑊𝑄 . Next, we use this historical information along with our persistent memory parameters as the input
sequence to the attention module:
˜S(𝑡 ) = 𝑝1 𝑝2 . . . 𝑝𝑁𝑝
 || ℎ𝑡 || S(𝑡 ) , (22)
𝑦𝑡 = Attn
 ˜S(𝑡 ) 
. (23)
The structure of the attention map over the entire sequence is shown in Figure 3a. We then use 𝑦𝑡 to update the long-term
memory module for the next segment and the final output:
M𝑡 = M𝑡 −1 (𝑦𝑡 ) , (24)
𝑜𝑡 = 𝑦𝑡 ⊗ M∗
𝑡 (𝑦𝑡 ) . (25)
Note that, in the above, we are updating the weight of M𝑡 −1 through forward pass.
This architecture has two key advantages: (1) Attention by having both historical and current context, has the ability to
decides whether given the current data, the long-term memory information is needed. (2) The attention module helps
9
Figure 4: Memory as a Gate (MAG) Architecture. This architecture, similarly, has the three branches of (1) core, (2)
contextual memory, and (3) persistent memory. It, however, incorporates only persistent memory into the context and
combine memory with the core branch using a gating mechanism. At test time, the behavior is the same as Figure 2.
the long-term memory to store only useful information from the current context. That is, not all tokens in each segment
are useful and memorizing all of them can result in memory overflow. Therefore, attention is helping the memory to
understand which information is useful, better managing the memory capacity. (3) At test time: (i) persistent memory
parameters are fixed as they encodes the knowledge about the task, which should not be changed; (ii) the attention module
weights are in-context learner; and (iii) the long-term memory module is still learning (memorizing) the information at test
time. That is, we update the weights of the neural memory even at test time as weights are encoding the abstraction of
long past.
4.2 Gated Memory
In the next variant (see Figure 4), in one branch, we directly use the input data to update the long-term memory, and in the
second branch, we use a sliding window attention (SWA):
˜𝑥 = 𝑝1 𝑝2 . . . 𝑝𝑁𝑝
 || 𝑥, (26)
𝑦 = SW-Attn∗ ( ˜𝑥) , (27)
𝑜 = 𝑦 ⊗ M ( ˜𝑥), (28)
where SW-Attn∗ is sliding window attention with prefix (see Figure 3b). Note that, contrary to the previous design, we are
not segmenting the input data. Also, we abuse the notation and use M (𝑥) to refer to the final output of the memory after
all recursion over the tokens of the sequence. In the above equation, ⊗ can be any non-linear gating. In our experiments,
we normalize the outputs 𝑦 and M ( ˜𝑥) using learnable vector-valued weights, followed by a non-linearity 𝜎 (.).
The overall attention mask of this design is shown in Figure 3b. In this design, sliding window attention is act as a precise
short-term memory, while the neural memory module is acting as a fading memory for the model. This architecture design
can also be seen as a multi-head architecture where the structure of heads are different (X. Dong et al. 2024).
4.3 Memory as a Layer
The last variant uses the neural Memory As a Layer (MAL) of a deep neural network (see Figure 5). This architecture
design is more common in the literature, where the hybrid models stack recurrent models with full or sliding window
attentions. Given input 𝑥, we have:
˜𝑥 = 𝑝1 𝑝2 . . . 𝑝𝑁𝑝
 || 𝑥, (29)
𝑦 = M ( ˜𝑥), (30)
𝑜 = SW-Attn (𝑦) , (31)
10
Figure 5: Memory as a Layer (MAL) Architecture. In this architecture, the memory layer is responsible to compress the
past and current context before the attention module.
where SW-Attn is sliding window attention. The main drawback of this design is that the power of the model is limited by
each of the layers and so it cannot take advantage of the complementary data processing of attention and neural memory
module. In our experiments, for evaluating memory in this design, we use a similar architecture as H3 (D. Y. Fu et al. 2023),
where we replace the the sequence model with our neural memory module (LMM).
Memory Without Attention. Although in the above, we discussed MAL as the combination of LMMs and attention in
a sequential manner, one simple variant of MAL is to treat LMM as a sequence model without any attention. From the
memory perspective, as discussed in Section 1, we expect each part of the memory system to work independently, even if
other components are disturbed. Therefore, a long-term memory module should still be a powerful model even without
short-term memory (i.e., attention). We refer to this variant as LMM or Titans (LMM) in our experiments. We provide
additional discussions on the connection of Titans and other modern recurrent models in Appendix C.
4.4 Architectural Details
For the sake of simplicity and presentation, we avoid discussing the implementation details like using residual connection,
gating with linear layer, and normalization. In all blocks, we use residual connections. In our implementation, we use
SiLU(.) activation (Elfwing, Uchibe, and Doya 2018) as the non-linear activation for computing query, key, and values and
normalize queries and keys using ℓ2-norm.
Convolution. Following the recent modern linear recurrent models (Gu and Dao 2024; S. Yang, Kautz, and Hatamizadeh
2024), we incorporate a 1D depthwise-separable convolution layer after each of the query, key, and value projections.
While not significantly affect the performance, these 1D convolutions have shown performance improvement and are also
computationally efficient.
Gating. We also follow the recent architectures that use normalization and gating with a linear layer before the final
output projection (Mehta et al. 2023).
Theorem 4.1. Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to TC0 (Merrill,
Petty, and Sabharwal 2024), Titans are capable of solving problems beyond TC 0, meaning that Titans are theoretically more
expressive than Transformers and most modern linear recurrent models in state tracking tasks.
5 Experiments
Next, we evaluate the performance of Titans and its variants in language modeling, commonsense reasoning, needle
in haystack, DNA modeling, and time series forecasting tasks1. In more details, in this section, we answer the
following empirical questions: (1) How do Titans perform compared to baselines in downstream tasks? (see §5.2,
1In the first version of the work, we aim to provide insights/evidences about why the learning paradigms of Titans are effective. We are working on
finalizing the results of larger models and will report them in the next version.
11
§5.6, and §5.7); (2) What is the actual context length of Titans? (see §5.3 and §5.4); (3) How do Titans scale with respect to
context length? (see §5.8); (4) How the depth of memory can affect both performance and efficiency? (see §5.5); and (5)
What is the contribution of each Titans’ component in its performance? (see §5.9).
5.1 Experimental Setup
Models. In our experiments, we focus on the three variants of Titans, which we refer to as: Titans with (1) Memory as a
Context (MAC), (2) Memory as a Gate (MAG), and (3) Memory as a Layer (MAL) as well as (4) neural memory module
alone. The reason behind using our long-term memory as a separate module is based on our definition of learning. As
discussed in Section 1, we define learning a process for acquiring effective and useful memory. Accordingly, we expect our
long-term memory to effectively learn from data, even without attention. For each of these models, we consider four scales
with: (i) 170M, (ii) 340M, (iii) 400M, and (iv) 760M parameters. While the first three are trained on 15B tokens sampled
from FineWeb-Edu dataset (Penedo et al. 2024), the last one is trained on 30B tokens from the same dataset.
Baselines. We compare our models with the state-of-the-art linear recurrent models, Transformers, and hybrid models
(recurrent + attention). More specifically in language tasks, we compare with Transformer++ (Touvron et al. 2023),
RetNet (Yutao Sun et al. 2023), Gated Linear Attention (GLA) (S. Yang, B. Wang, Shen, et al. 2024), Mamba (Gu and Dao
2024), Mamba2 (Dao and Gu 2024), DeltaNet (S. Yang, B. Wang, Yu Zhang, et al. 2024), TTT (Yu Sun et al. 2024), and Gated
DeltaNet (S. Yang, Kautz, and Hatamizadeh 2024). In needle in haystack tasks, we also compare with GPT4 (Achiam et al.
2023), Llama3 with RAG (Touvron et al. 2023), RecurrentGemma2-9B (Botev et al. 2024), and Mistral (Jiang et al. 2023)
models, all of which are provided in the benchmark (Yuri Kuratov et al. 2024). In time series tasks, we compare with
Mamba-based (Behrouz, Santacatterina, and Zabih 2024), Transformer-based (Y. Liu et al. 2023; Nie et al. 2022; Yunhao
Zhang and Yan 2023), and linear models (Das et al. 2023; Z. Li et al. 2023; H. Wu et al. 2023; Zeng et al. 2023).
Training. In the training, we follow the training procedure of S. Yang, Kautz, and Hatamizadeh (2024), and use LLama 2
tokenizer with a vocabulary size of 32K and use training length of 4K tokens. We employ AdamW optimizer with learning
rate of 4𝑒-4 with cosine annealing schedule with batch size of 0.5M tokens, and weight decay of 0.1.
5.2 Language Modeling
We first focus on the perplexity in language modeling and also commonsense reasoning tasks. The results for Titans’
variants and also baselines with three different sizes of 340M, 400M, and 760M are reported in Table 1. Among non-hybrid
models, including Transformer++, our neural memory module achieves the best performance in both perplexity and
accuracy measures. Comparing our neural memory module and TTT, which is also a gradient-based recurrent model can
show us the importance of our weight decay as well as the momentum. As discussed earlier, the weight decay can be
interpreted as a gating mechanism to forget the past data, when it is needed. Also, momentum can help us better manage
the memory by providing additional memory for the surprise metric. While some baselines also take advantage of gating
mechanism, e.g., Mamba, Mamba2, and Gated DeltaNet, the superior performance of our neural memory module shows
the importance of both our surprise mechanism and having deep and non-linear memory. We further discuss the later in
Section 5.5.
Comparing the hybrid models, we found that all three variants of Titans (MAC, MAG, and MAL) outperform both Samba
(Mamba + attention) and Gated DeltaNet-H2 (Gated DeltaNet + atttention). We attribute the superior performance of Titans
(MAL) to the power of neural memory module as the architecture design and used attention are all the same. Comparing
Titans (MAG) and (MAC), we find that while their performance are close, MAC performs better when dealing with longer
dependencies in the data. Interestingly, both MAG and MAC outperform MAL variant, which due to using the same
modules, we attribute this to the architecture design of these models. This finding is particularly important as the current
hybrid models (except Hymba (X. Dong et al. 2024)) in the literature are using MAL-style combination of recurrent models
and attention.
5.3 Needle in a Haystack
Scaling a model to longer context window is not always equivalent to being effective for very long sequences (Hsieh
et al. 2024). The needle-in-a-haystack (NIAH) task is designed to measure the actual effective context length of models.
In this task, we evaluate the model on retrieving a piece of information (i.e., the “needle”) from long distractor texts (i.e.,
12
Table 1: Performance of Titans and recurrent- and Transformer-based baselines on language modeling and common-sense
reasoning tasks. Hybrid models are marked with ∗. The best results among simple and hybrid models are highlighted.
Model Wiki. LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c SIQA BoolQ Avg.
ppl ↓ ppl ↓ acc ↑ acc ↑ acc_n ↑ acc ↑ acc ↑ acc_n ↑ acc ↑ acc ↑ ↑
340M params / 15B tokens
Transformer++ 31.52 41.08 30.76 62.98 34.76 50.53 45.21 24.05 36.81 58.24 42.92
RetNet 32.50 49.73 28.24 62.61 34.15 50.91 44.27 23.62 36.79 59.72 42.54
GLA 28.51 43.02 28.73 64.05 35.96 50.00 54.19 24.29 37.13 58.39 44.09
Mamba 30.83 40.21 29.94 63.79 35.88 49.82 49.24 24.56 35.41 60.07 43.59
DeltaNet 28.65 47.30 28.43 63.52 35.95 49.63 52.68 25.37 37.96 58.79 44.04
TTT 27.44 34.19 30.06 63.97 35.71 50.08 53.01 26.11 37.32 59.83 44.51
Gated DeltaNet 27.01 30.94 34.11 63.08 38.12 51.60 55.28 26.77 34.89 59.54 45.42
Titans (LMM) 26.18 29.97 34.98 64.73 39.61 51.85 55.60 28.14 34.52 59.99 46.17
Titans (MAC)∗ 25.43 28.13 36.00 65.32 40.35 51.21 58.17 29.00 38.63 60.18 47.36
Titans (MAG)∗ 25.07 28.72 36.71 64.88 40.56 52.49 57.72 28.16 39.75 60.01 47.54
Titans (MAL)∗ 24.69 28.80 35.74 64.97 39.44 51.97 56.58 28.21 38.14 57.32 46.55
400M params / 15B tokens
Transformer++ 30.63 37.37 29.64 64.27 37.72 51.53 54.95 27.36 38.07 61.59 45.64
RetNet 29.92 46.83 29.16 65.23 36.97 51.85 56.01 27.55 37.30 59.66 45.47
HGRN2 32.33 47.14 26.12 64.52 35.45 52.24 55.97 25.51 37.35 59.02 44.52
GLA 27.96 36.66 27.86 65.94 37.41 49.56 56.01 26.36 38.94 59.84 45.24
Mamba 29.22 39.88 29.82 65.72 37.93 50.11 58.37 26.70 37.76 61.13 45.94
Mamba2 26.34 33.19 32.03 65.77 39.73 52.48 59.00 27.64 37.92 60.72 46.91
DeltaNet 27.69 44.04 29.96 64.52 37.03 50.82 56.77 27.13 38.22 60.09 45.57
TTT 26.11 31.52 33.25 65.70 39.11 51.68 58.04 28.99 38.26 59.87 46.86
Gated DeltaNet 25.47 29.24 34.40 65.94 40.46 51.46 59.80 28.58 37.43 60.03 47.26
Samba∗ 25.32 29.47 36.86 66.09 39.24 51.45 60.12 27.20 38.68 58.22 47.23
Gated DeltaNet-H2∗ 24.19 28.09 36.77 66.43 40.79 52.17 59.55 29.09 39.04 58.56 47.69
Titans (LMM) 25.03 28.99 35.21 65.85 40.91 52.19 59.97 29.20 38.74 60.85 47.83
Titans (MAC)∗ 25.61 27.73 36.92 66.39 41.18 52.80 60.24 29.69 40.07 61.93 48.65
Titans (MAG)∗ 23.59 27.81 37.24 66.80 40.92 53.21 60.01 29.45 39.91 61.28 48.60
Titans (MAL)∗ 23.93 27.89 36.84 66.29 40.74 52.26 59.85 29.71 38.92 58.40 47.87
760M params / 30B tokens
Transformer++ 25.21 27.64 35.78 66.92 42.19 51.95 60.38 32.46 39.51 60.37 48.69
RetNet 26.08 24.45 34.51 67.19 41.63 52.09 63.17 32.78 38.36 57.92 48.46
Mamba 28.12 23.96 32.80 66.04 39.15 52.38 61.49 30.34 37.96 57.62 47.22
Mamba2 22.94 28.37 33.54 67.90 42.71 49.77 63.48 31.09 40.06 58.15 48.34
DeltaNet 24.37 24.60 37.06 66.93 41.98 50.65 64.87 31.39 39.88 59.02 48.97
TTT 24.17 23.51 34.74 67.25 43.92 50.99 64.53 33.81 40.16 59.58 47.32
Gated DeltaNet 21.18 22.09 35.54 68.01 44.95 50.73 66.87 33.09 39.21 59.14 49.69
Samba∗ 20.63 22.71 39.72 69.19 47.35 52.01 66.92 33.20 38.98 61.24 51.08
Gated DeltaNet-H2∗ 19.88 20.83 39.18 68.95 48.22 52.57 67.01 35.49 39.39 61.11 51.49
Titans (LMM) 20.04 21.96 37.40 69.28 48.46 52.27 66.31 35.84 40.13 62.76 51.56
Titans (MAC) 19.93 20.12 39.62 70.46 49.01 53.18 67.86 36.01 41.87 62.05 52.51
Titans (MAG) 18.61 19.86 40.98 70.25 48.94 52.89 68.23 36.19 40.38 62.11 52.50
Titans (MAL) 19.07 20.33 40.05 69.99 48.82 53.02 67.54 35.65 30.98 61.72 50.97
the “haystack”). In this part, we use Single NIAH (S-NIAH) task from RULER benchmark (Hsieh et al. 2024) and evaluate
Titans and baselines on sequences with length 2K, 4K, 8K, and 16K. The results are reported in Table 2. Neural Memory
module achieves the best results compare to baselines in all three tasks. We attribute this superior performance to three
key differences of Titans with existing sequence models: (1) Compared to TTT, our Neural Memory can better handle the
memory capacity by using momentum and also the forgetting mechanism (i.e., weight decay). Therefore, with increasing
the sequence length, the performance of Neural Memory does not drop and show a consistent trend; (2) Compared to
Mamba2, which has the gating (forgetting) mechanism, Titans have deep non-linear memory, resulting in better memory
management. Also, contrary to our neural memory and DeltaNet, Mamba2 is not capable of removing a memory and so
13
Table 2: Performance of Titans and baselines on S-NIAH task from RULER benchmark. The best results among simple
and hybrid models are highlighted.
Model S-NIAH-PK S-NIAH-N S-NIAH-W
2K 4K 8K 16K 2K 4K 8K 16K 2K 4K 8K 16K
TTT 98.4 98.8 98.0 88.4 60.2 36.6 10.2 4.4 78.8 28.0 4.4 0.0
Mamba2 98.6 61.4 31.0 5.4 98.4 55.8 14.2 0.0 42.2 4.2 0.0 0.0
DeltaNet 96.8 98.8 98.6 71.4 47.2 15.4 12.8 5.4 46.2 20.0 1.6 0.0
Titans (LMM) 99.8 98.4 98.2 96.2 100.0 99.8 93.4 80.2 90.4 89.4 85.8 80.6
Titans (MAC) 99.2 98.8 99.0 98.4 99.6 98.2 97.6 97.4 98.2 98.2 95.6 95.2
Titans (MAG) 99.4 98.0 97.4 97.4 99.2 98.8 97.2 98.6 98.0 98.0 90.2 88.2
Titans (MAL) 98.8 98.6 98.8 97.8 99.8 98.1 96.8 96.4 98.0 97.4 92.0 90.4
(a) Few-shot Setup (b) Fine-Tuning Setup
Figure 6: Performance of Titans and baselines on BABILong benchmark. Titans (MAC) outperforms all baselines, including
extremely large models, e.g., GPT4.
we can see a significant drop in performance when increasing the sequence length; (3) Compared to DeltaNet, although it
is capable of removing memory using delta rule, it cannot erase the memory, lacking forgetting mechanism. Finally, As
expected we can see on par or better results when using Titans variants, where the best results correspond to MAC.
5.4 BABILong Benchmark
In the previous section we discussed the results on a simple NIAH tasks where a single needle needs to be retrieved.
Although Titans showed better performance compared to baselines, their true advantage over very long sequences is still
hidden. To this end, in this section, we use a harder task from BABILong benchmark (Yuri Kuratov et al. 2024), in which
the model needs to reason across facts distributed in extremely long documents. We follow the original experimental setup
and training process in the benchmark. There are two settings: (1) Few-shot setting, in which we use large pre-trained
models, and (2) fine-tuning setting, where we fine-tune the MAC variant of Titans to compare it with other fine-tuned
baselines. The results for few-shot setting are reported in Figure 6a. In this setup, we can see Titans outperform all
baselines–i.e., Mamba2.8B (Gu and Dao 2024), RWKV-6-7B (Peng, Goldstein, et al. 2024), RecurrentGemma-9B (Botev et al.
2024), Gemma-9B (Team et al. 2024), Llama3.1-8B (Touvron et al. 2023), GPT-4, and GPT4o-mini (Achiam et al. 2023). These
results are achieved while Titans (MAC) is having much less number of parameters than baselines.
In the fine-tuning setup, we compare the small fine-tuned version of Titans (MAC) with: (i) the fine-tuned version of small
models (almost the same number of parameters as Titans) such as Mamba (Gu and Dao 2024), RMT (Bulatov, Yury Kuratov,
and Burtsev 2022), (ii) large models with Retrieval-Augmented Generation (RAG) (P. Lewis et al. 2020) such as Llama3.1-
8B (Touvron et al. 2023), and (iii) extremely large models such as GPT-4 (Achiam et al. 2023), GPT4o-mini, Qwen2.5-72B (A.
Yang et al. 2024), and Llama3.1-70B (Touvron et al. 2023). Baseline results are reported by (Yuri Kuratov et al. 2024). The
results of Titans and baselines are reported in Figure 6b. Titans outperform all models even extremely large models like
GPT4. Also, compared to Transformer-based with memory models like RMT, Titans show better performance mainly due
to their powerful memory. That is, RMT compress the historical data into 16 size vector-valued memory, while Titans with
in-context online memory learner are capable of encoding the past into the parameters of the model. Interestingly, even
14
(a) 170M Parameters (b) 360M Parameters (c) 760M Parameters
Figure 7: The effect of memory depth on the perplexity. Deeper long-term memory results in better scaling in longer
sequences.
Table 3: Performance on long-term forecasting. The best results are highlighted .
Neural Memory Simba iTransformer RLinear PatchTST Crossformer TiDE TimesNet DLinear
MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
ETTm1 0.358 0.387 0.383 0.396 0.407 0.410 0.414 0.407 0.387 0.400 0.513 0.496 0.419 0.419 0.400 0.406 0.403 0.407
ETTm2 0.261 0.309 0.271 0.327 0.288 0.332 0.286 0.327 0.281 0.326 0.757 0.610 0.358 0.404 0.291 0.333 0.350 0.401
ETTh1 0.420 0.421 0.441 0.432 0.454 0.447 0.446 0.434 0.469 0.454 0.529 0.522 0.541 0.507 0.458 0.450 0.456 0.452
ETTh2 0.336 0.382 0.361 0.391 0.383 0.407 0.374 0.398 0.387 0.407 0.942 0.684 0.611 0.550 0.414 0.427 0.559 0.515
ECL 0.162 0.261 0.169 0.274 0.178 0.270 0.219 0.298 0.205 0.290 0.244 0.334 0.251 0.344 0.192 0.295 0.212 0.300
Traffic 0.415 0.289 0.493 0.291 0.428 0.282 0.626 0.378 0.481 0.304 0.550 0.304 0.760 0.473 0.620 0.336 0.625 0.383
Weather 0.231 0.265 0.255 0.280 0.258 0.278 0.272 0.291 0.259 0.281 0.259 0.315 0.271 0.320 0.259 0.287 0.265 0.317
augmenting Llama3.1-8B model with RAG performs worse than Titans with about ×70 less parameters.
5.5 The Effect of Deep Memory
In this section, we evaluate the effect of deep memory in both wall-clock training time and model performance2. To this
end, we focus on different variants of our neural memory module, where 𝐿M = 1, 2, 3, 4. We also use Mamba as a baseline
for the model performance. For a fair comparison, we use the same training process for all models and train them on a
subset of the Pile dataset (L. Gao et al. 2020).
We report the perplexity of our models and baselines as the function of the sequence length in Figure 7. Interestingly, with
the increase of memory depth, 𝐿M , the model can achieve better perplexity over all sequence length. Also, deeper memory
modules are more robust to the sequence length when the model has less number of parameters. With the increase of the
number of parameters, all models show better performance on longer sequences.
Figure 8: The effect of memory depth on
training throughput
We also evaluate the effect of memory depth (𝐿M = 1, 2, 3, 4) on the training
throughput. We report the training throughput (the number of tokens per
second) as the function of sequence length in Figure 8. All models scale linearly
with respect to the context length (i.e., constant trend in the number of tokens
per second with respect to sequence length). Also, by increasing the memory
depth, as expected, we can see a linear trend that a deeper memory results in
a slower training. Therefore, it is not always efficient to use deeper memory
modules, showing a trade-off between effectiveness and efficiency.
5.6 Time Series Forecasting
To show the effectiveness of our memory module in a broader tasks, we also evaluate its performance in time series
forecasting tasks. To this end, we use Simba framework (Patro and Agneeswaran 2024) for time series forecasting, and
2Note that, in this experiment, we only focus on the neural memory module to evaluate the effect of memory depth in the memorization process.
Combining neural memory with attention as we do in Titans variants, can additionally enhance the performance of the model over long sequences.
15
Table 4: Downstream evaluation of pre-trained DNA models on GenomicsBenchmarks (Grešová et al. 2023). We report
top-1 classification accuracy (%).
Model Enhancer Cohn Enhancer Ens Human Reg. Non-TATA Promoters Human OCR Ens.
CNN 69.5 68.9 93.3 84.6 68.0
DNABERT 74.0 85.7 88.1 85.6 75.1
GPT 70.5 83.5 91.5 87.7 73.0
HyenaDNA 74.2 89.2 93.8 96.6 80.9
Transformer++ 73.4 89.5 89.9 94.4 79.5
Mamba 73.0 - - 96.6 -
Based 74.6 89.5 89.5 96.8 79.0
Neural Memory Module 75.2 89.6 89.3 96.6 79.9
replace its Mamba module with our neural memory. We report the results on common time series forecasting benchmark
datasets–ETT, ECL, Traffic, and Weather (H. Zhou et al. 2021). The results are reported in Table 3. Our neural memory
module is outperforming all baselines, including Mamba-based, linear-based, and Transformer-based architectures.
5.7 DNA Modeling
In order to understand the capability of Titans beyond natural language, we further evaluate the performance of our
neural memory module on DNA modeling tasks. To this end, we evaluate pre-trained models on the downstream tasks
in GenomicsBenchmarks (Grešová et al. 2023). We follow the same experimental setups from Nguyen et al. (2024), and
re-use the reported results of baselines by Arora et al. (2024). The performance of Titans (LMM) and baselines are reported
in Table 4. We find that LMM is competitive with state-of-the-art architectures across different downstream genomics
tasks.
5.8 Efficiency
Figure 9: Training throughput compari-
son of Titans and baselines.
In this part, we compare the efficiency of our neural memory as well as Titans
with state-of-the-art sequence models. The training throughput of models for
different sequence length × batch size are reported in Figure 9. Comparing
recurrent models, including our neural memory module, we can see our memory
module is slightly slower than Mamba2 and Gated DeltaNet, mainly due to: (1)
having deep memory and more expressive transition process (memory update),
and (2) highly optimized kernel in the implementation of Mamba2. Interestingly,
Titans (MAL) are faster than baselines as well as the memory module. The
main reason for this better throughput is the highly optimized kernel of Flash-
Attention (Dao 2024), which is used for implementing SWA and full attention
module in Titans.
5.9 Ablation Study
Finally, we perform ablation studies on the different architectural choices in Titans. We consider our neural memory
module as a base model and then changing one component at a time: (1) replacing deep memory with linear memory,
removing (2) convolution, (3) momentum in the surprise measure, (4) weight decay (or forgot mechanism), and (5) persistent
memory. The results are reported in Table 5. All components of neural memory design are positively contributing to its
performance, where the greatest contribution comes from weight decay, momentum, convolution, and persistent memory,
respectively.
The Effect of Architectural Design. To evaluate the effect of architecture design, we compare the performance of three
represented variants of Titans in three aspects of (i) language modeling, (ii) commen-sense reasoning, and (iii) long context
NIAH (BABILong) tasks. The results are reported in Table 5. We find that MAC and MAG have close performance in
language modeling and common-sense reasoning tasks, while MAC achieve significantly better performance in long-context
NIAH. Both of these models achieve better performance than MAL. These results along with Figure 9, show a trade-off
between fast training and more expressive design.
16
Table 5: Ablation Study on Titans. All components of Titans are positively contributing to its performance.
Model Language Modeling Reasoning Long Context
ppl ↓ acc ↑ acc ↑
LMM 27.01 47.83 92.68
+Attn (MAC) 26.67 48.65 97.95
+Attn (MAG) 25.70 48.60 96.70
+Attn (MAL) 25.91 47.87 96.91
Linear Memory 28.49 46.97 85.34
w/o Convolution 28.73 45.82 90.28
w/o Momentum 28.98 45.49 87.12
w/o Weight Decay 29.04 45.11 85.60
w/o Persistent Memory 27.63 46.35 92.49
6 Conclusion
In this paper, we present a neural long-term memory that, as a meta in-context learner, learns to memorize at test time.
The neural memory module is a recurrent model in nature, and is adaptively memorizing tokens that are more surprising
or are close to surprising tokens. Comparing to modern recurrent models, it has more expressive memory update and
storing mechanism. Using this memory, we present Titans architectures, and its three variants, in which we suggest to
incorporate the memory module as (1) a context, (2) gating, and (3) a layer. Our experimental evaluation on diverse tasks
tasks validate that Titans are more effective than Transformers and recent modern linear recurrent models, specifically for
long context. That is, Titans can scale to larger than 2M context window size with better accuracy than baselines.
Titans are implemented in Pytorch and JAX and we intend to make the code we used to train and evaluate our models
available soon.
17


A Related Work
There are diverse perspectives that can independently lead to the design of Titans or its components. Accordingly, to
further situate our work in a broader context, we review three categories of studies:

A.1 Linear Recurrent Models
Recently, to address the computational cost of Transformers in both training and inference, linear recurrent models
have attracted much attention (Tiezzi et al. 2024), mainly due to their fast inference and training. The first generation
of models–such as RetNet (Yutao Sun et al. 2023), LRU (Orvieto et al. 2023), RWKV (Peng, Alcaide, et al. 2023), S5 (J. T.
Smith, Warrington, and Linderman 2023), and S4 (Gu, Goel, and Re 2022)–uses data-independent transition matrix/decay
mechanism. The second generation of such models started to incorporate gating mechanism, a widely used techniques
in traditional RNNs (Gers, Jürgen Schmidhuber, and Cummins 2000; Greff et al. 2016; Van Der Westhuizen and Lasenby
2018), into such linear architectures–e.g., Griffin (De et al. 2024), SSMs (Behrouz, Santacatterina, and Zabih 2024; Dao
and Gu 2024; Gu and Dao 2024; Hasani et al. 2023), RWKV6 (Peng, Goldstein, et al. 2024). The third generation of linear
recurrent models are based on more complex memory updating rule based on meta-learning, online learning, and/or
delta-rule, resulting in more expressive and effective models such as: Longhorn (B. Liu et al. 2024), Gated DeltaNet (S. Yang,
Kautz, and Hatamizadeh 2024), TTT (Yu Sun et al. 2024), and DeltaNet (S. Yang, B. Wang, Yu Zhang, et al. 2024). Our
LMM model can be seen as the next generation of such models, in which we incorporate the token flow into the memory
updating mechanism, having more powerful memory updating process. See Appendix C for a detailed discussion of
different recurrent models and Titans.

A.2 Transformer-based Architectures
Transformers. Transformers (Vaswani et al. 2017) as the de facto backbone for many deep learning models are based on
attention mechanism (Bahdanau 2014). They, however, suffer from quadratic computational cost, limiting their ability
to scale to long context window. To improve the memory consumption and throughput of softmax attention for longer
sequences, various studies focused on I/O aware implementations of attention (Dao 2024; Dao, D. Fu, et al. 2022), designing
more efficient attention mechanisms by sparsifying the attention matrix (B. Chen et al. 2021; Choromanski et al. 2021; Dai
et al. 2019; J. Dong et al. 2024; Roy et al. 2021), approximating the softmax (Arora et al. 2024), or developing kernel-based
(linear) attentions (Aksenov et al. 2024; Kacham, Mirrokni, and P. Zhong 2024; Schlag, Irie, and Jürgen Schmidhuber 2021;
S. Yang, B. Wang, Shen, et al. 2024).
Segment-based Transformers. Another line of research to improve the efficiency of Transformers is segment-based or
Chunk Transformers (Dai et al. 2019). The main drawback of chunk Transformers is that segments are fully separated and
so the context window is limited to the length of the chunks. To address this issue, various studies discuss the importance
of a memory so it can help the model to transfer information across chunks (Bulatov, Yuri Kuratov, et al. 2023; Bulatov,
Yury Kuratov, and Burtsev 2022; Feng et al. 2022; Hutchins et al. 2022; Rodkin et al. 2024; Z. Wang et al. 2019; Q. Wu
et al. 2020; Zancato et al. 2024). The key differences of Titans with these models are: (1) The memory in such models are
simple small size vectors, lacking expressive power to compress complex information; (2) The memory module lacks forget
mechanism, leading to a fast memory overflow; (3) only focus on momentary surprise, missing the information flow. More
specifically, recalling Recurrent Memory Transformers (RMT) (Bulatov, Yuri Kuratov, et al. 2023; Bulatov, Yury Kuratov,
and Burtsev 2022; Rodkin et al. 2024), one can treat Titans (MAC) as the generalization of RMT, where we use a neural
memory module instead of a vector-valued small size memory.
Memory for Large Language Models. Another interesting research direction has been to incorporate external memory
modules to LLMs after training (Z. He et al. 2024; Khandelwal et al. 2020; Y. Wang, Y. Gao, et al. 2024). Such models
are different from our approach as we incorporate the memory as a part of initial architecture and so we train it in
an end-to-end manner. Also, most of these explicit memory modules suffer from the same limitations as chunk-based
Transformers (mentioned above). For a detailed discussion of such models, we refer to the recent study of Y. Wang, Han,
et al. (2024).
25
A.3 Test Time Training and Fast Weight Programs
Memory Design and Augmentation with Memory. In the literature, a substantial research effort have been toward
designing memory modules that are capable of either memorizing the knowledge abstraction (e.g., persistent mem-
ory) (Sukhbaatar, Grave, et al. 2019), or memorizing the data-dependent information (also known as contextual memory),
through recurrence (Bulatov, Yury Kuratov, and Burtsev 2022; Rodkin et al. 2024; Zancato et al. 2024), Transformers (Berges
et al. 2024; Cetin et al. 2024; Feng et al. 2022; Le, Tran, and Venkatesh 2020; Munkhdalai, Faruqui, and Gopal 2024; J. Zhang
et al. 2024), gradient (Irie, Csordás, and Jürgen Schmidhuber 2022; Munkhdalai, Sordoni, et al. 2019), or other learning
paradigms (Sukhbaatar, Weston, Fergus, et al. 2015; Weston, Chopra, and Bordes 2014). These memory models, however,
either (1) are based on momentary surprise, missing the data flow and events, (2) lack forget mechanisms to remove
the memory, leading to a fast memory overflow (3) are fixed-size shallow (matrix valued) memory, resulting in poor
performance in long context, and (4) are based on fixed parameters at test time, lacking test time adaption.
Fast Weight Programs. The idea of seeing linear layers as the key-value (associative) memory system backs to fast
weight programs, in which dynamic fast programs are incorporated into recurrent neural networks to serve as writable
memory (Schlag, Irie, and Jürgen Schmidhuber 2021; JH Schmidhuber 1992; Jürgen Schmidhuber 1993). The two learning
rules of Hebbian (Hebb 2005) and delta (Prados and Kak 1989) are the most popular learning rules for fast weight programs,
which have been extensively explored in various studies (Irie, Schlag, et al. 2021; Munkhdalai, Sordoni, et al. 2019;
Munkhdalai and H. Yu 2017; Schlag, Irie, and Jürgen Schmidhuber 2021; JH Schmidhuber 1992; S. Yang, Kautz, and
Hatamizadeh 2024; S. Yang, B. Wang, Yu Zhang, et al. 2024). All these models, however, are based on momentary surprise,
missing the token flow in the sequences (see Section 3.1), and most of them lacks a forgetting gate, resulting in a poor
memory management.
Test Time Training. The key ideas of learning at test time or learning to learn (i.e., (Andrychowicz et al. 2016)) backs to
very early studies on local learning Bottou and Vapnik 1992, in which each test data sample is trained on its neighbors
before making a prediction (Gandelsman et al. 2022; H. Zhang et al. 2006). This approach further has shown promising
performance in vision tasks (Jain and Learned-Miller 2011; Mullapudi et al. 2019), mostly due to their ability to mitigate
out-of-distribution samples. The most similar studies to ours in this direction are MNM (Munkhdalai, Sordoni, et al. 2019)
and TTT-layer (Yu Sun et al. 2024), which we discussed the key differences in Appendix C.
B Language Modeling and Common-sense Reasoning Datasets
Following recent studies on linear recurrent models (Dao and Gu 2024; S. Yang, Kautz, and Hatamizadeh 2024; S. Yang,
B. Wang, Yu Zhang, et al. 2024), we use Wikitext (Merity et al. 2017), LMB (Paperno et al. 2016), PIQA (Bisk et al. 2020),
HellaSwag (Zellers et al. 2019), WinoGrande (Sakaguchi et al. 2021), ARC-easy (ARC-e) and ARC-challenge (ARC-c) (P.
Clark et al. 2018), SIQA (Sap et al. 2019), and BoolQ (C. Clark et al. 2019). Also, the baselines results for 400M models are
from the reported results by S. Yang, Kautz, and Hatamizadeh (2024).
C Long-term Memory Module (LMM) as a Sequence Model
In this section, we discuss how LMM as a sequence model is connected to modern linear recurrent models. For the sake
of simplicity, we start with a linear memory, where M𝑡 = 𝑊𝑡 ∈ R𝑑in ×𝑑in . In this case, our objective function becomes
ℓ (M; 𝑥𝑡 ) = 1
2 ∥M𝑡 k𝑡 − v𝑡 ∥2
2, in which we use gradient descent with momentum and weight decay for the optimization.
Accordingly, revisiting the recurrent formula in Equation 13:
M𝑡 = diag (1 − 𝛼𝑡 ) M𝑡 + 𝑆𝑡 (32)
𝑆𝑡 = diag (𝜂𝑡 ) 𝑆𝑡 −1 − diag (𝜃𝑡 ) M𝑡 −1k⊤
𝑡 k𝑡 − v⊤
𝑡 k𝑡
 . (33)
LMM is Generalized Gated DeltaNet. As discussed by S. Yang, Kautz, and Hatamizadeh (2024), DeltaNet (S. Yang, B. Wang,
Yu Zhang, et al. 2024) can alternatively be interpreted as an online learning problem that optimizes the L = 1
2 ∥S𝑡 k𝑡 − v𝑡 ∥2
2,
resulting in:
S𝑡+1 = S𝑡 − 𝜃𝑡 ∇L = S𝑡
I − 𝜃𝑡 k𝑡 k⊤
𝑡
 + 𝜃𝑡 v𝑡 k⊤
𝑡 . (34)
26
In this formulation, Gated DeltaNet is the same as above but with an additional weight decay term (S. Yang, Kautz, and
Hatamizadeh 2024). Comparing Equation 32 and Equation 34, we can see that setting 𝜂𝑡 = 0 results in both formulations to
be equivalent. Accordingly, we can say LMM is generalizing the very recent study of Gated DeltaNet (S. Yang, Kautz, and
Hatamizadeh 2024) from three aspects:
• Momentum-based Rule: The Delta Rule is based on momentary surprise, meaning that the flow of tokens cannot
affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and
momentary surprise.
• Deep Memory: While Gated DeltaNet is limited to a linear (matrix-valued) memory as it requires finding the closed
recurrence form, LMM allows using deep memory module by using a gradient-based formulation, resulting in higher
expressive power.
• Non-Linear Recurrence : While DeltaNet and Gated DeltaNet are based on linear recurrence, our LMM is using
inter-chunk non-linear recurrence and intra-chunk linear recurrence. This design allows LMM having a higher
expressive power.
Here, we discussed Gated DeltaNet as a sample of recent generation of recurrent models. Similar approaches such
as RWKV-7 (Peng 2021) are also using the same formulation and loss function, and so LMM is generalizing all such
models.
LMM is Generalized Longhorn. Similar to DeltaNet, Longhorn (B. Liu et al. 2024) uses the same loss function but it
derives the closed form using implicit online learning:
S𝑡+1 = S𝑡
I − 𝛿𝑡 k𝑡 k⊤
𝑡
 + 𝛿𝑡 v𝑡 k⊤
𝑡 , (35)
where 𝛿𝑡 = 𝜃𝑡
1+𝜃𝑡 k𝑡 k⊤
𝑡
. It, however, lacks a forgetting gate, resulting in a faster memory overflow. Therefore, in addition two
the abovementioned aspects of (1) Momentum-based Rule , (2) Deep Memory , and (3) Non-Linear Recurrence , LMM has
the advantage of using an additional (4) Forget Gate, leading to a better memory management.
LMM is Generalized TTT Layer. To the best of our knowledge, TTT (Yu Sun et al. 2024), is the only modern linear
recurrent models with a gradient-based updating rule. In addition to different architectural designs and also objective
functions, our LMM has three key differences with presented TTT layers (Yu Sun et al. 2024):
1. Forgetting Mechanism: TTT layers are updating memory at each time, without having the chance to forget the
past data. Accordingly, when fixing the memory size, the model cannot manage the memory for long sequences. A
forget mechanism, such as LMM’s, allows clearing the memory when very past information is not needed anymore.
We show that in a general case, this forget mechanism is equivalent to weight decay and provide a fast method to
incorporate it into the parallel training.
2. Momentum-based Update Rule : TTT layers are based on momentary surprise, meaning that the flow of tokens
cannot affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and
momentary surprise. See Section 3.1 for the motivation of this design.
3. Deep Memory : While TTT-layers allows for deeper memory, the advantages/disadvantages of such deeper memory
modules have not been experimentally evaluated.
To the best of our knowledge, our neural long-term memory module is the first linear recurrent model with momentum-
based update rule.
Finally, as a key difference with all the above and other recent linear recurrent studies, note that the hybrid variants of
modern linear models–such as Griffin (De et al. 2024), DeltaNet (S. Yang, B. Wang, Yu Zhang, et al. 2024), Gated DeltaNet (S.
Yang, Kautz, and Hatamizadeh 2024), H3 (D. Y. Fu et al. 2023), Mamba2 (Dao and Gu 2024), Samba (Ren et al. 2024), etc.–all
are based on sequential layer-wise design. We present Titans to show how effectively one can incorporate such memory
modules into an architecture.
27

In this formulation, Gated DeltaNet is the same as above but with an additional weight decay term (S. Yang, Kautz, and
Hatamizadeh 2024). Comparing Equation 32 and Equation 34, we can see that setting 𝜂𝑡 = 0 results in both formulations to
be equivalent. Accordingly, we can say LMM is generalizing the very recent study of Gated DeltaNet (S. Yang, Kautz, and
Hatamizadeh 2024) from three aspects:
• Momentum-based Rule: The Delta Rule is based on momentary surprise, meaning that the flow of tokens cannot
affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and
momentary surprise.
• Deep Memory: While Gated DeltaNet is limited to a linear (matrix-valued) memory as it requires finding the closed
recurrence form, LMM allows using deep memory module by using a gradient-based formulation, resulting in higher
expressive power.
• Non-Linear Recurrence : While DeltaNet and Gated DeltaNet are based on linear recurrence, our LMM is using
inter-chunk non-linear recurrence and intra-chunk linear recurrence. This design allows LMM having a higher
expressive power.
Here, we discussed Gated DeltaNet as a sample of recent generation of recurrent models. Similar approaches such
as RWKV-7 (Peng 2021) are also using the same formulation and loss function, and so LMM is generalizing all such
models.
LMM is Generalized Longhorn. Similar to DeltaNet, Longhorn (B. Liu et al. 2024) uses the same loss function but it
derives the closed form using implicit online learning:
S𝑡+1 = S𝑡
I − 𝛿𝑡 k𝑡 k⊤
𝑡
 + 𝛿𝑡 v𝑡 k⊤
𝑡 , (35)
where 𝛿𝑡 = 𝜃𝑡
1+𝜃𝑡 k𝑡 k⊤
𝑡
. It, however, lacks a forgetting gate, resulting in a faster memory overflow. Therefore, in addition two
the abovementioned aspects of (1) Momentum-based Rule , (2) Deep Memory , and (3) Non-Linear Recurrence , LMM has
the advantage of using an additional (4) Forget Gate, leading to a better memory management.
LMM is Generalized TTT Layer. To the best of our knowledge, TTT (Yu Sun et al. 2024), is the only modern linear
recurrent models with a gradient-based updating rule. In addition to different architectural designs and also objective
functions, our LMM has three key differences with presented TTT layers (Yu Sun et al. 2024):
1. Forgetting Mechanism: TTT layers are updating memory at each time, without having the chance to forget the
past data. Accordingly, when fixing the memory size, the model cannot manage the memory for long sequences. A
forget mechanism, such as LMM’s, allows clearing the memory when very past information is not needed anymore.
We show that in a general case, this forget mechanism is equivalent to weight decay and provide a fast method to
incorporate it into the parallel training.
2. Momentum-based Update Rule : TTT layers are based on momentary surprise, meaning that the flow of tokens
cannot affect the memory update rule. LMM, however, is based on a momentum rule, which consider both past and
momentary surprise. See Section 3.1 for the motivation of this design.
3. Deep Memory : While TTT-layers allows for deeper memory, the advantages/disadvantages of such deeper memory
modules have not been experimentally evaluated.
To the best of our knowledge, our neural long-term memory module is the first linear recurrent model with momentum-
based update rule.
Finally, as a key difference with all the above and other recent linear recurrent studies, note that the hybrid variants of
modern linear models–such as Griffin (De et al. 2024), DeltaNet (S. Yang, B. Wang, Yu Zhang, et al. 2024), Gated DeltaNet (S.
Yang, Kautz, and Hatamizadeh 2024), H3 (D. Y. Fu et al. 2023), Mamba2 (Dao and Gu 2024), Samba (Ren et al. 2024), etc.–all
are based on sequential layer-wise design. We present Titans to show how effectively one can incorporate such memory
modules into an architecture.

Memory-Augmented Attention Mechanisms
The paper includes the following equations, formulas, functions, and algorithms:
•
Equation 1: Defines how query (Q), key (K), and value (V) matrices are calculated from input x using learnable parameters WQ, WK, and WV [1].
◦
Q = xWQ, K = xWK, V = xWV
•
Equation 2: Defines how the output yᵢ of the attention mechanism is calculated, based on the softmax of the dot product between query and key matrices, and value matrices [1].
◦
yᵢ = (∑ⱼ=₁ⁱ exp(QᵢᵀKⱼ/√dᵢₙ) Vⱼ) / (∑ₗ=₁ⁱ exp(QᵢᵀKₗ/√dᵢₙ))
•
Equation 3: Describes linear attention, where the softmax in standard attention is replaced with a kernel function φ(., .), where φ(x, y) = φ(x)φ(y) [2].
◦
yᵢ = (∑ⱼ=₁ⁱ φ(QᵢᵀKⱼ)Vⱼ) / (∑ₗ=₁ⁱ φ(QᵢᵀKₗ)) = (∑ⱼ=₁ⁱ φ(Qᵢ)ᵀφ(Kⱼ)Vⱼ) / (∑ₗ=₁ⁱ φ(Qᵢ)ᵀφ(Kₗ)) = φ(Qᵢ)ᵀ (∑ⱼ=₁ⁱ φ(Kⱼ)Vⱼ) / φ(Qᵢ)ᵀ (∑ₗ=₁ⁱ φ(Kₗ))
•
Equation 4: Shows a recurrent format for linear attention using a memory matrix M [3].
◦
Mₜ = Mₜ₋₁ + KₜᵀVₜ
•
Equation 5: Shows the calculation of the output yₜ using the query Qₜ and memory Mₜ [3].
◦
yₜ = QₜMₜ
•
Equations 6 and 7: Show the general form of a recurrent neural network (RNN) with a memory unit M, where f(.,.) is the write operation and g(.,.) is the read operation [4].
◦
Mₜ = f(Mₜ₋₁, xₜ), Write Operation
◦
yₜ = g(Mₜ, xₜ), Read Operation
•
Equation 8: Defines the initial update rule for the long-term memory M based on the gradient of a loss function ℓ with respect to the input xₜ [5].
◦
Mₜ = Mₜ₋₁ - θₜ ∇ℓ(Mₜ₋₁; xₜ)
•
Equation 9: Defines an updated memory update rule by adding a surprise metric Sₜ [6].
◦
Mₜ = Mₜ₋₁ + Sₜ
•
Equation 10: Defines the surprise metric Sₜ, which is a combination of past surprise and momentary surprise, using data-dependent surprise decay (ηₜ) and a term that controls how much momentary surprise is incorporated (θₜ) [6].
◦
Sₜ = ηₜSₜ₋₁ - θₜ∇ℓ(Mₜ₋₁; xₜ)
•
Equation 11: Shows how input xₜ is projected into key kₜ and value vₜ using linear layers with learnable parameters Wₖ and Wv [7].
◦
kₜ = xₜWₖ, vₜ = xₜWv
•
Equation 12: Defines the loss function ℓ used to train the memory module, which is based on the squared difference between the memory's output and the value vector [7].
◦
ℓ(Mₜ₋₁; xₜ) = ||Mₜ₋₁(kₜ) - vₜ||²₂
•
Equation 13: Modifies the memory update rule with a forgetting mechanism using a gating parameter αₜ [8].
◦
Mₜ = (1 - αₜ)Mₜ₋₁ + Sₜ
•
Equation 14: Restates the calculation of the surprise metric for the modified memory update rule with a forgetting mechanism [8].
•
Sₜ = ηₜSₜ₋₁ - θₜ ∇ℓ(Mₜ₋₁; xₜ)
•
Equation 15: Shows how to retrieve a memory, yₜ, by passing the query vector qₜ through the memory module without weight adjustment [9].
◦
yₜ = M*(qₜ)
•
Equation 16: Represents the weights in the inner loop with mini-batch gradient descent, data-dependent learning rate, and weight decay [10].
•
Mₜ = (1 - αₜ)Mₜ₋₁ - θₜ ∇ℓ(Mₜ₋₁; xₜ) = βₜM₀ - ∑ᵢ=₁ᵗ (θᵢ/βₜ)βᵢ ∇ℓ(Mₜ'; xᵢ)
•
Equation 17: Shows the mini-batch gradient descent with weight decay for a linear memory [11].
◦
∇ℓ(W₀; xₜ) = (W₀xₜ - xₜ)xₜᵀ ⇒ ∑ᵢ=₁ᵇ (θᵢ/βᵇ)βᵢ ∇ℓ(W₀; xᵢ) = ΘᵇBᵇ(W₀X - X)Xᵀ
•
Equation 18: Shows the momentum term in a chunk-wise gradient descent [12].
◦
Sₜ = ηₜSₜ₋₁ - θₜuₜ
•
Equation 19: Modifies the input by adding learnable persistent memory parameters P at the beginning of the sequence [13].
◦
xnew = [p₁ p₂ ... pNp] || x
•
Equation 20: Shows the similarity between fully connected layers in Transformer architectures and attention weights with input-independent parameters [14].
◦
FFN(x) = Wv Softmax(Wₖx)
•
Equation 21: Defines how the past information, hₜ, is retrieved from the long-term memory M using a query vector qₜ [15].
◦
hₜ = M*ₜ₋₁(qₜ)
•
Equation 22: Defines the new input segment by concatenating persistent memory parameters, retrieved long-term memory, and current segment [16].
◦
S̃(ₜ) = [p₁ p₂ ... pNp] || hₜ || S(ₜ)
•
Equation 23: Applies attention to the modified segment [16].
◦
yₜ = Attn(S̃(ₜ))
•
Equation 24: Shows how the long-term memory is updated using the output of the attention module [16].
◦
Mₜ = Mₜ₋₁(yₜ)
•
Equation 25: Calculates the final output by combining the attention output with the long-term memory output [16].
◦
oₜ = yₜ ⊗ M*ₜ(yₜ)
•
Equation 26: Modifies the input by adding persistent memory parameters [17].
◦
x̃ = [p₁ p₂ ... pNp] || x
•
Equation 27: Applies sliding window attention (SWA) to the input [17].
◦
y = SW-Attn*(x̃)
•
Equation 28: Combines sliding window attention output and memory output using a gating mechanism [17].
◦
o = y ⊗ M(x̃)
•
Equation 29: Modifies the input by adding persistent memory parameters [18].
◦
x̃ = [p₁ p₂ ... pNp] || x
•
Equation 30: Passes the modified input through the memory module [18].
◦
y = M(x̃)
•
Equation 31: Applies sliding window attention (SWA) to the output of the memory module [18].
◦
o = SW-Attn(y)
•
Equation 32: Restates the recurrent formula for the memory module with weight decay [19].
◦
Mₜ = diag(1 - αₜ)Mₜ + Sₜ
•
Equation 33: Restates the surprise metric for a linear memory module [19].
◦
Sₜ = diag(ηₜ)Sₜ₋₁ - diag(θₜ)(Mₜ₋₁kₜᵀkₜ - vₜᵀkₜ)
•
Equation 34: Represents the DeltaNet update rule for the memory module [19].
◦
Sₜ₊₁ = Sₜ - θₜ∇L = Sₜ(I - θₜkₜkₜᵀ) + θₜvₜkₜᵀ
•
Equation 35: Represents the Longhorn update rule for the memory module [20].
◦
Sₜ₊₁ = Sₜ(I - δₜkₜkₜᵀ) + δₜvₜkₜᵀ

**Equation 1**
* **Name**: _Linear Projection for Attention_
* **Purpose**: To create the Query (Q), Key (K), and Value (V) matrices, which are essential components for the attention mechanism.
* **Inputs**: Input matrix `x`
* **Arguments/Parameters**: Learnable weight matrices `WQ`, `WK`, and `WV`.
* **Logical Flow/Sequence**: The input matrix `x` is multiplied by each of the learnable weight matrices `WQ`, `WK`, and `WV` respectively, to generate the query (Q), key (K), and value (V) matrices.
* **Outputs**: Query matrix `Q`, Key matrix `K`, and Value matrix `V`.
* **Limitations**: This equation represents a linear transformation.
* **Advantages**: It is a simple and computationally efficient method to generate the Q, K, and V matrices from the input. It introduces learnable parameters, which allows the model to learn the optimal transformations.

**Equation 2**
* **Name**: _Standard Attention Calculation_
* **Purpose**: To calculate the output `yᵢ` of the attention mechanism, representing the weighted sum of value vectors based on the relevance of each input position to the current query position.
* **Inputs**: Query matrix `Q`, Key matrix `K`, and Value matrix `V`
* **Arguments/Parameters**: `dᵢₙ` is the dimension of the input.
* **Logical Flow/Sequence**:
* For each position `i`, the dot product of query vector `Qᵢ` with each key vector `Kⱼ` is calculated, and then divided by the square root of `dᵢₙ`.
* The exponential of this scaled dot product is computed.
* The result is then used as weights over the value vectors `Vⱼ` to create a weighted sum.
* Finally, the weighted sum is normalized by the sum of exponential scaled dot products.
* **Outputs**: Output vector `yᵢ`.
* **Limitations**: Has a quadratic complexity in terms of context length and is computationally expensive for long sequences.
* **Advantages**: Allows the model to attend to different parts of the input sequence and capture dependencies between the input elements.

**Equation 3**
* **Name**: _Linear Attention Calculation_
* **Purpose**: To compute the output `yᵢ` using a kernel function `φ(., .)` instead of the softmax used in standard attention, reducing computational complexity.
* **Inputs**: Query matrix `Q`, Key matrix `K`, and Value matrix `V`.
* **Arguments/Parameters**: `φ(., .)` is the kernel function, where `φ(x, y) = φ(x)φ(y)`.
* **Logical Flow/Sequence**:
* The dot product between `Qᵢ` and `Kⱼ` is computed using the kernel function `φ`.
* The kernelized dot product is used as a weight over the value vectors `Vⱼ` to form a weighted sum.
* The weighted sum is normalized by the sum of kernelized dot products.
* The kernel function is factorized into φ(Qᵢ) and φ(Kⱼ).
* **Outputs**: Output vector `yᵢ`
* **Limitations**: Can lead to a performance drop compared to standard attention.
* **Advantages**: Reduces the computational complexity from quadratic to linear in terms of sequence length, enabling the model to handle longer sequences.

**Equation 4**
* **Name**: _Recurrent Linear Attention Memory Update_
* **Purpose**: To define a recurrent update rule for a memory matrix `M`, used in linear attention mechanisms.
* **Inputs**: Key matrix `Kₜ` and Value matrix `Vₜ` at time step t, and memory matrix `Mₜ₋₁` at time step t-1.
* **Arguments/Parameters**: None
* **Logical Flow/Sequence**: The memory matrix `M` at time `t` is updated by adding the product of the transpose of key matrix `Kₜ` and value matrix `Vₜ` to the memory matrix from the previous time step `Mₜ₋₁`.
* **Outputs**: Updated memory matrix `Mₜ`.
* **Limitations**: The memory matrix is updated additively, which can lead to memory overflow when dealing with long sequences.
* **Advantages**: Allows for an efficient, recurrent way to compute the memory update.

**Equation 5**
* **Name**: _Linear Attention Output Calculation_
* **Purpose**: To calculate the output vector `yₜ` using the query matrix `Qₜ` and the updated memory matrix `Mₜ`.
* **Inputs**: Query matrix `Qₜ` and Memory matrix `Mₜ`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**: The output vector `yₜ` is calculated by taking the product of query matrix `Qₜ` and the memory matrix `Mₜ`.
* **Outputs**: Output vector `yₜ`.
* **Limitations**: The output is directly dependent on the accumulated information in the memory matrix `Mₜ`.
* **Advantages**: Computationally efficient, it allows the model to make predictions based on current query and historical context stored in the memory matrix.

**Equations 6 and 7**
* **Name**: _General Recurrent Neural Network (RNN) Formulation_
* **Purpose**: To define the general structure of a recurrent neural network with a memory unit `M`.
* **Inputs**: Memory unit `Mₜ₋₁` at time t-1, and input data `xₜ` at time t.
* **Arguments/Parameters**: `f(., .)` represents the write operation function, and `g(., .)` represents the read operation function.
* **Logical Flow/Sequence**:
* The memory unit `M` at time `t` is updated by the write operation function `f(Mₜ₋₁, xₜ)`, which takes the previous memory state and current input.
* The output `yₜ` is computed by applying the read operation function `g(Mₜ, xₜ)` to the current memory state and current input.
* **Outputs**: Updated memory matrix `Mₜ` and output vector `yₜ`.
* **Limitations**: The specific behavior depends on the choice of functions `f` and `g`.
* **Advantages**: This generalized representation can be used to describe many different types of recurrent models, including RNNs, LSTMs, and GRUs.

**Equation 8**
* **Name**: _Initial Memory Update with Gradient_
* **Purpose**: To define the initial update rule for long-term memory, where memory is updated based on the gradient of a loss function.
* **Inputs**: Long-term memory `Mₜ₋₁` at time t-1 and input data `xₜ` at time t.
* **Arguments/Parameters**: `θₜ` controls the update magnitude, and `∇ℓ(Mₜ₋₁; xₜ)` is the gradient of the loss function with respect to input `xₜ`.
* **Logical Flow/Sequence**:
* The gradient of the loss function `ℓ` with respect to the input `xₜ` is calculated using the current memory state `Mₜ₋₁`.
* This gradient is then scaled by a factor `θₜ`.
* The memory is updated by subtracting the scaled gradient from the previous memory state.
* **Outputs**: Updated long-term memory `Mₜ`.
* **Limitations**: Only considers the momentary surprise (gradient) of an input.
* **Advantages**: Allows the memory to be updated based on how surprising the current input is.

**Equation 9**
* **Name**: _Memory Update with Surprise Metric_
* **Purpose**: To define an updated memory update rule using a surprise metric `Sₜ`.
* **Inputs**: Long-term memory `Mₜ₋₁` and surprise metric `Sₜ`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**: The memory at time t, `Mₜ`, is updated by adding the surprise metric `Sₜ` to the previous memory state `Mₜ₋₁`.
* **Outputs**: Updated long-term memory `Mₜ`.
* **Limitations**: The surprise metric `Sₜ` is not yet fully defined in this equation.
* **Advantages**: Provides a general form for memory update that can incorporate a more complex notion of surprise.

**Equation 10**
* **Name**: _Surprise Metric Calculation_
* **Purpose**: To define the surprise metric `Sₜ` as a combination of past surprise and momentary surprise.
* **Inputs**: Past surprise metric `Sₜ₋₁` and long-term memory `Mₜ₋₁` at time t-1, and input data `xₜ` at time t.
* **Arguments/Parameters**: `ηₜ` is the data-dependent surprise decay, `θₜ` controls how much momentary surprise is incorporated, and `∇ℓ(Mₜ₋₁; xₜ)` is the gradient of the loss function with respect to input `xₜ`.
* **Logical Flow/Sequence**:
* The past surprise `Sₜ₋₁` is decayed by the factor `ηₜ`.
* The gradient of the loss function with respect to the input `xₜ` is calculated using the current memory state `Mₜ₋₁`.
* This gradient, representing the momentary surprise, is scaled by `θₜ`.
* The surprise metric `Sₜ` is computed by subtracting the scaled gradient from the decayed past surprise.
* **Outputs**: Surprise metric `Sₜ`.
* **Limitations**: The effectiveness depends on the appropriate choice of data-dependent parameters `ηₜ` and `θₜ`.
* **Advantages**: This formulation is similar to gradient descent with momentum and incorporates both past and momentary surprise.

**Equation 11**
* **Name**: _Linear Projection for Key and Value_
* **Purpose**: To project the input `xₜ` into key `kₜ` and value `vₜ` vectors using learnable linear layers.
* **Inputs**: Input data `xₜ`.
* **Arguments/Parameters**: Learnable weight matrices `Wₖ` and `Wv`.
* **Logical Flow/Sequence**: The input `xₜ` is multiplied by weight matrix `Wₖ` to obtain the key vector `kₜ` and by `Wv` to get the value vector `vₜ`.
* **Outputs**: Key vector `kₜ` and Value vector `vₜ`.
* **Limitations**: The key and value vectors are linear projections of the input.
* **Advantages**: Provides a simple and efficient way to create the key and value vectors.

**Equation 12**
* **Name**: _Associative Memory Loss Function_
* **Purpose**: To define the loss function `ℓ` used to train the memory module, which measures the difference between the memory's output and the value vector.
* **Inputs**: Long-term memory `Mₜ₋₁`, key vector `kₜ`, and value vector `vₜ`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**:
* The memory module's output `Mₜ₋₁(kₜ)` is calculated given key vector `kₜ`.
* The squared L2 norm (Euclidean distance) between the memory module's output and value vector `vₜ` is computed.
* **Outputs**: Loss value `ℓ(Mₜ₋₁; xₜ)`.
* **Limitations**: The loss function is based on the assumption of an associative relationship between keys and values.
* **Advantages**: Encourages the memory module to learn the mapping between keys and values at test time.

**Equation 13**
* **Name**: _Memory Update with Forgetting Mechanism_
* **Purpose**: To modify the memory update rule with a forgetting mechanism controlled by the gating parameter `αₜ`.
* **Inputs**: Long-term memory `Mₜ₋₁` at time t-1, surprise metric `Sₜ`, and forgetting gate `αₜ`.
* **Arguments/Parameters**: `αₜ` is the gating parameter, which ranges from 0 to 1.
* **Logical Flow/Sequence**:
* The previous memory `Mₜ₋₁` is multiplied by `(1 - αₜ)`. This is the forgetting mechanism.
* The surprise metric `Sₜ` is added to this result.
* **Outputs**: Updated long-term memory `Mₜ`.
* **Limitations**: The effectiveness depends on the proper value of the gating parameter `αₜ`.
* **Advantages**: Provides a mechanism to forget less important information, thus managing memory capacity and avoiding memory overflow.

**Equation 14**
* **Name**: _Surprise Metric Calculation with Forgetting Mechanism_
* **Purpose**: To restate the surprise metric `Sₜ` calculation for the modified memory update rule with a forgetting mechanism.
* **Inputs**: Past surprise metric `Sₜ₋₁` and long-term memory `Mₜ₋₁` at time t-1 and input data `xₜ` at time t.
* **Arguments/Parameters**: `ηₜ` is the data-dependent surprise decay, `θₜ` controls how much momentary surprise is incorporated, and `∇ℓ(Mₜ₋₁; xₜ)` is the gradient of the loss function with respect to input `xₜ`.
* **Logical Flow/Sequence**: The same as **Equation 10**, the surprise metric `Sₜ` is computed by decaying the past surprise and subtracting the scaled momentary surprise from the decayed past surprise.
* **Outputs**: Surprise metric `Sₜ`.
* **Limitations**: It's the same as the limitations of **Equation 10**.
* **Advantages**: It's the same as the advantages of **Equation 10**.

**Equation 15**
* **Name**: _Memory Retrieval_
* **Purpose**: To retrieve a memory `yₜ` by passing a query vector `qₜ` through the memory module without weight adjustment.
* **Inputs**: Query vector `qₜ` and the memory module `M`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**:
* The query vector `qₜ` is passed through the memory module `M` without updating any of the weights, to get the corresponding memory `yₜ`.
* **Outputs**: Retrieved memory `yₜ`.
* **Limitations**: The retrieved memory is based on the current state of the memory module.
* **Advantages**: Provides a direct way to access information stored in the memory.

**Equation 16**
* **Name**: _Mini-Batch Gradient Descent with Weight Decay_
* **Purpose**: To represent the weights in the inner loop using mini-batch gradient descent, data-dependent learning rate, and weight decay.
* **Inputs**: Long-term memory at the previous step `Mₜ₋₁`, input data `xₜ` at time t.
* **Arguments/Parameters**: `αₜ` is the gating parameter, `θᵢ` is the data dependent learning rate, `βₜ` is a term related to the decay, `∇ℓ(Mₜ'; xᵢ)` is the gradient of the loss function with respect to input `xᵢ`.
* **Logical Flow/Sequence**:
* The weights of the memory module are updated with mini-batch gradient descent, taking into account the weight decay term and the data-dependent learning rate.
* This is done by accumulating the gradients across the mini-batch.
* **Outputs**: Updated long-term memory `Mₜ`.
* **Limitations**: The effectiveness depends on the values of hyperparameters such as the learning rate and weight decay rate.
* **Advantages**: Allows for parallelizable training of the memory module.

**Equation 17**
* **Name**: _Mini-Batch Gradient Descent with Weight Decay for Linear Memory_
* **Purpose**: To show how mini-batch gradient descent with weight decay is calculated for a linear memory, using matmul operations.
* **Inputs**: Initial linear memory `W₀`, input data `xₜ` at time t.
* **Arguments/Parameters**: `θᵢ` is the data dependent learning rate, `βᵢ` is a term related to the decay. `X` is a matrix containing all input samples of a chunk, `Θᵇ` is a diagonal matrix of the data dependent learning rates, and `Bᵇ` is a matrix defined on `βᵢ`.
* **Logical Flow/Sequence**:
* The gradient of the loss function is computed for a linear memory matrix `W₀` with respect to the input `xₜ`.
* The summation of these gradients across the mini-batch is expressed using matmul operations.
* **Outputs**: The accumulated gradient for a chunk of data, expressed in a way that allows for fast parallel calculation.
* **Limitations**: It is specifically for a linear memory.
* **Advantages**: The calculation is structured so that it is highly parallelizable, using matmuls.

**Equation 18**
* **Name**: _Momentum Term in Chunk-Wise Gradient Descent_
* **Purpose**: To define the momentum term in a chunk-wise gradient descent, which helps accelerate learning by considering past gradients.
* **Inputs**: Previous momentum `Sₜ₋₁` and current gradient `uₜ`.
* **Arguments/Parameters**: `ηₜ` is the data-dependent surprise decay, `θₜ` controls how much of the current gradient is incorporated.
* **Logical Flow/Sequence**: The momentum term `Sₜ` is calculated by decaying the previous momentum `Sₜ₋₁` with `ηₜ` and subtracting the current gradient `uₜ` scaled by `θₜ`.
* **Outputs**: Momentum term `Sₜ`.
* **Limitations**: The behavior of the momentum depends on the values of `ηₜ` and `θₜ`.
* **Advantages**: Helps to speed up the convergence by accumulating the gradients across steps.

**Equation 19**
* **Name**: _Input Modification with Persistent Memory Parameters_
* **Purpose**: To modify the input sequence by adding learnable, persistent memory parameters `P` at the beginning of the sequence.
* **Inputs**: Input sequence `x`.
* **Arguments/Parameters**: `p₁`, `p₂`, ..., `pNp` are learnable persistent memory parameters.
* **Logical Flow/Sequence**: The learnable parameters `p₁`, `p₂`, ..., `pNp` are concatenated to the beginning of the input sequence `x`, resulting in a new input `xnew`.
* **Outputs**: Modified input sequence `xnew`.
* **Limitations**: Adds a fixed number of parameters to the input sequence.
* **Advantages**: Allows the model to learn task-related information that is independent of the input sequence.

**Equation 20**
* **Name**: _Similarity Between Fully Connected Layers and Attention_
* **Purpose**: To demonstrate the similarity between fully connected layers in Transformer architectures and attention weights with input-independent parameters.
* **Inputs**: Input data `x`.
* **Arguments/Parameters**: Weight matrices `Wv` and `Wₖ`.
* **Logical Flow/Sequence**:
* The input `x` is multiplied by a weight matrix `Wₖ`.
* The softmax function is applied to the result.
* This softmax output is then multiplied by another weight matrix `Wv`.
* **Outputs**: Output of the fully connected layer, `FFN(x)`.
* **Limitations**: This equation shows the similarity, not the equivalence between the FFN and attention.
* **Advantages**: Highlights the fact that data-independent parameters in FFN can act like attention weights.

**Equation 21**
* **Name**: _Long-Term Memory Retrieval (MAC)_
* **Purpose**: To define how past information `hₜ` is retrieved from the long-term memory `M` using a query vector `qₜ` in the Memory as a Context (MAC) architecture.
* **Inputs**: Long-term memory module `M*ₜ₋₁` and query vector `qₜ`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**:
* The query vector `qₜ` is passed through the memory module `M*ₜ₋₁`, which is the memory from the last segment, without weight adjustments.
* **Outputs**: Retrieved past information `hₜ`.
* **Limitations**: The information is retrieved based on the memory state `M*ₜ₋₁`.
* **Advantages**: It's a way to use past learned memory and incorporate it with the current input.

**Equation 22**
* **Name**: _Segment Modification (MAC)_
* **Purpose**: To define a new input segment by concatenating persistent memory parameters, retrieved long-term memory, and current segment in the Memory as a Context (MAC) architecture.
* **Inputs**: Persistent memory parameters `p₁`...`pNp`, retrieved long-term memory `hₜ`, and current segment `S(ₜ)`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**: The persistent memory parameters, retrieved long-term memory and current segment are concatenated together in that order.
* **Outputs**: Modified input segment `S̃(ₜ)`.
* **Limitations**: The order of concatenation is fixed.
* **Advantages**: Creates a combined context for the attention mechanism, using the current input with long-term and task-specific information.

**Equation 23**
* **Name**: _Attention Application (MAC)_
* **Purpose**: To apply the attention mechanism to the modified segment, incorporating both current and long-term memory in the Memory as a Context (MAC) architecture.
* **Inputs**: Modified segment `S̃(ₜ)`.
* **Arguments/Parameters**: `Attn` represents the attention mechanism.
* **Logical Flow/Sequence**: The modified segment `S̃(ₜ)` is passed through the attention mechanism `Attn`.
* **Outputs**: Attention output `yₜ`.
* **Limitations**: The output is dependent on the attention module's implementation.
* **Advantages**: Allows the attention to decide which part of the input (persistent, historical, or current) should be used.

**Equation 24**
* **Name**: _Long-Term Memory Update (MAC)_
* **Purpose**: To define how the long-term memory is updated using the output of the attention module `yₜ` in the Memory as a Context (MAC) architecture.
* **Inputs**: Previous long-term memory `Mₜ₋₁` and attention output `yₜ`.
* **Arguments/Parameters**: None.
* **Logical Flow/Sequence**: The previous long-term memory `Mₜ₋₁` is updated through the memory module by passing the output of the attention module `yₜ` into it.
* **Outputs**: Updated long-term memory `Mₜ`.
* **Limitations**: The memory is updated using the output of the attention module, which is the combined context.
* **Advantages**: The long-term memory stores useful information from the current context using attention.

**Equation 25**
* **Name**: _Final Output Calculation (MAC)_
* **Purpose**: To calculate the final output `oₜ` by combining the attention output with the long-term memory output in the Memory as a Context (MAC) architecture.
* **Inputs**: Attention output `yₜ` and long-term memory output from the current time step `M*ₜ(yₜ)`.
* **Arguments/Parameters**: `⊗` represents a non-linear combination of attention output and memory output.
* **Logical Flow/Sequence**: The final output `oₜ` is obtained by combining the attention output `yₜ` with the output of the long-term memory when the attention output is passed through it `M*ₜ(yₜ)`.
* **Outputs**: Final output `oₜ`.
* **Limitations**: The effectiveness depends on the choice of the non-linear combination operator `⊗`.
* **Advantages**: Allows the model to combine the short-term context from attention with long-term information from memory.

**Equation 26**
* **Name**: _Input Modification with Persistent Memory Parameters (MAG)_
* **Purpose**: To modify the input sequence by adding persistent memory parameters at the beginning, similar to **Equation 19** but used in Memory as a Gate (MAG) architecture.
* **Inputs**: Input sequence `x`.
* **Arguments/Parameters**: `p₁`, `p₂`, ..., `pNp` are learnable persistent memory parameters.
* **Logical Flow/Sequence**: The learnable parameters `p₁`, `p₂`, ..., `pNp` are concatenated to the beginning of the input sequence `x`, resulting in a new input `x̃`.
* **Outputs**: Modified input sequence `x̃`.
* **Limitations**: The same as **Equation 19**.
* **Advantages**: The same as **Equation 19**.

**Equation 27**
* **Name**: _Sliding Window Attention Application (MAG)_
* **Purpose**: To apply sliding window attention (SWA) to the modified input in the Memory as a Gate (MAG) architecture.
* **Inputs**: Modified input sequence `x̃`.
* **Arguments/Parameters**: `SW-Attn*` represents the sliding window attention mechanism.
* **Logical Flow/Sequence**: The modified input `x̃` is processed using the sliding window attention mechanism `SW-Attn*`.
* **Outputs**: Sliding window attention output `y`.
* **Limitations**: The output depends on the implementation of the sliding window attention.
* **Advantages**: Efficient method of applying attention on long sequences by using a sliding window.

**Equation 28**
* **Name**: _Final Output Calculation with Gating (MAG)_
* **Purpose**: To combine the sliding window attention output and memory output using a gating mechanism in the Memory as a Gate (MAG) architecture.
* **Inputs**: Sliding window attention output `y`, memory module output `M(x̃)`.
* **Arguments/Parameters**: `⊗` represents a non-linear gating mechanism.
* **Logical Flow/Sequence**: The final output `o` is obtained by combining the sliding window attention output `y` with the memory output `M(x̃)` using the gating mechanism `⊗`.
* **Outputs**: Final output `o`.
* **Limitations**: The result is dependent on the choice of the gating function `⊗`.
* **Advantages**: It allows the model to combine the short-term attention output with long-term memory by gating.

**Equation 29**
* **Name**: _Input Modification with Persistent Memory Parameters (MAL)_
* **Purpose**: To modify the input sequence by adding persistent memory parameters at the beginning in the Memory as a Layer (MAL) architecture.
* **Inputs**: Input sequence `x`.
* **Arguments/Parameters**: `p₁`, `p₂`, ..., `pNp` are learnable persistent memory parameters.
* **Logical Flow/Sequence**: Similar to equation 19 and 26, the learnable parameters `p₁`, `p₂`, ..., `pNp` are concatenated to the beginning of the input sequence `x`, resulting in a new input `x̃`.
* **Outputs**: Modified input sequence `x̃`.
* **Limitations**: The same as **Equation 19** and **Equation 26**.
* **Advantages**: The same as **Equation 19** and **Equation 26**.

**Equation 30**
* **Name**: _Memory Module Application (MAL)_
* **Purpose**: To pass the modified input through the memory module in the Memory as a Layer (MAL) architecture.
* **Inputs**: Modified input sequence `x̃`.
* **Arguments/Parameters**: `M` represents the memory module.
* **Logical Flow/Sequence**: The modified input `x̃` is passed through the memory module `M`.
* **Outputs**: Output of memory module `y`.
* **Limitations**: The output is entirely dependent on the functionality of the memory module `M`.
* **Advantages**: Allows the memory module to process the input before it goes through the attention mechanism.

**Equation 31**
* **Name**: _Sliding Window Attention Application (MAL)_
* **Purpose**: To apply sliding window attention (SWA) to the output of the memory module in the Memory as a Layer (MAL) architecture.
* **Inputs**: Output of the memory module `y`.
* **Arguments/Parameters**: `SW-Attn` represents the sliding window attention mechanism.
* **Logical Flow/Sequence**: The output of the memory module `y` is processed using the sliding window attention mechanism `SW-Attn`.
* **Outputs**: Sliding window attention output `o`.
* **Limitations**: The output depends on the implementation of the sliding window attention.
* **Advantages**: Applies attention after the input is processed by the memory module.

**Equation 32**
* **Name**: _Recurrent Memory Update with Weight Decay_
* **Purpose**: To restate the recurrent formula for the memory module with a weight decay mechanism using a diagonal matrix.
* **Inputs**: Memory matrix `Mₜ` and surprise metric `Sₜ`.
* **Arguments/Parameters**: `αₜ` is the gating parameter, which ranges from 0 to 1.
* **Logical Flow/Sequence**:
* The memory matrix `Mₜ` is multiplied by a diagonal matrix `diag(1 - αₜ)` where the diagonal contains the `1 - αₜ` values.
* The surprise metric `Sₜ` is added to the decayed memory matrix.
* **Outputs**: Updated memory `Mₜ`.
* **Limitations**: The weight decay is controlled by the diagonal matrix `diag(1 - αₜ)`.
* **Advantages**: Provides a mechanism for forgetting less important past information, managing memory capacity and avoiding memory overflow.

**Equation 33**
* **Name**: _Surprise Metric for Linear Memory_
* **Purpose**: To restate the surprise metric `Sₜ` for a linear memory module, incorporating data-dependent terms `ηₜ`, `θₜ`, and linear projections.
* **Inputs**: Previous surprise metric `Sₜ₋₁`, memory matrix `Mₜ₋₁`, key vector `kₜ`, and value vector `vₜ`.
* **Arguments/Parameters**: `ηₜ` is a data-dependent surprise decay and `θₜ` controls how much momentary surprise is incorporated, both as diagonal matrices.
* **Logical Flow/Sequence**:
* The previous surprise metric `Sₜ₋₁` is multiplied by a diagonal matrix `diag(ηₜ)`.
* The term `Mₜ₋₁kₜᵀkₜ - vₜᵀkₜ` is multiplied by a diagonal matrix `diag(θₜ)`.
* The second term is subtracted from the first term.
* **Outputs**: Updated surprise metric `Sₜ`.
* **Limitations**: This is specifically designed for linear memory, meaning that `M` is a matrix and not a multi-layer network.
* **Advantages**: Extends the surprise metric to linear memory using data-dependent decay and weights.

**Equation 34**
* **Name**: _DeltaNet Update Rule_
* **Purpose**: To represent the DeltaNet update rule for the memory module, showing how it modifies memory based on the gradient of a loss function.
* **Inputs**: Memory `Sₜ`, learning rate `θₜ`, key vector `kₜ`, and value vector `vₜ`.
* **Arguments/Parameters**: `θₜ` is the learning rate, `∇L` is the gradient of the loss function.
* **Logical Flow/Sequence**:
* The memory `Sₜ` is updated by subtracting `θₜ` multiplied by the gradient of the loss function.
* The gradient term is calculated as `Sₜkₜkₜᵀ - θₜvₜkₜᵀ` which can be simplified and written as `Sₜ(I - θₜkₜkₜᵀ) + θₜvₜkₜᵀ`.
* **Outputs**: Updated memory `Sₜ₊₁`.
* **Limitations**: This is a simplified update rule for linear memory.
* **Advantages**: Based on the delta rule, a fast method for updating the memory.

**Equation 35**
* **Name**: _Longhorn Update Rule_
* **Purpose**: To represent the Longhorn update rule for the memory module, showing an alternative method to the DeltaNet to modify the memory based on the gradient.
* **Inputs**: Memory `Sₜ`, learning rate `δₜ`, key vector `kₜ`, and value vector `vₜ`.
* **Arguments/Parameters**: `δₜ` is the learning rate, derived from `θₜ`.
* **Logical Flow/Sequence**:
* The memory `Sₜ` is updated using the learning rate and a function of the key `kₜ` and value `vₜ` vectors.
* The update is similar to the DeltaNet rule, but with a different learning rate.
* **Outputs**: Updated memory `Sₜ₊₁`.
* **Limitations**: This is a simplified update rule for linear memory.
* **Advantages**: Based on a implicit online learning method.



### Equation (1): Linear Projections for Attention
**Equation:** 
```plaintext
Q = xW_Q, K = xW_K, V = xW_V
```
**Name:** Linear Projections for Attention

**Purpose:** 
To compute the Query (Q), Key (K), and Value (V) matrices from the input sequence using learnable weight matrices, which are essential components in the attention mechanism.

**Inputs/Parameters:** 
- `x`: Input matrix of shape (N, d_in), where N is sequence length and d_in is the input feature dimension.
- `W_Q`, `W_K`, `W_V`: Learnable weight matrices of shape (d_in, d_in).

**Logical Flow/Sequence:** 
1. Multiply the input `x` by the weight matrix `W_Q` to get the query matrix `Q`.
2. Multiply the input `x` by the weight matrix `W_K` to get the key matrix `K`.
3. Multiply the input `x` by the weight matrix `W_V` to get the value matrix `V`.

**Outputs:** 
- `Q`, `K`, `V`: Matrices each of shape (N, d_in) used in subsequent attention computations.

**Limitations:** 
- Assumes input and output dimensions match (square weight matrices), may need adjustments for different dimensions.
- Does not include non-linear activation; purely linear projections.

**Advantages:** 
- Simple and efficient computation.
- Forms the basis for the attention mechanism, allowing subsequent operations to leverage these projections.

---

### Equation (2): Causal Attention Output
**Equation:** 
```plaintext
y_i = (∑_{j=1}^i exp(Q_i^T K_j / √d_in) V_j) / (∑_{l=1}^i exp(Q_i^T K_l / √d_in))
```
**Name:** Causal Attention Output

**Purpose:** 
To compute the output of an attention mechanism at position i, considering only the past and current tokens (causal or autoregressive setup).

**Inputs/Parameters:** 
- `Q_i`: The i-th row of the Query matrix.
- `K_j`: The j-th row of the Key matrix for j ≤ i.
- `V_j`: The j-th row of the Value matrix for j ≤ i.
- `d_in`: Dimension of input/feature vector (used for normalization in softmax).

**Logical Flow/Sequence:** 
1. For each position `i`, compute the dot product between query `Q_i` and each key `K_j` for `j` from 1 to `i`.
2. Scale each dot product by `1/√d_in`.
3. Apply the exponential function to these scaled dot products.
4. Compute a weighted sum of the value vectors `V_j`, where weights are the exponentiated dot products normalized by the sum over all positions `l` from 1 to `i`.
5. The result is the output vector `y_i` at position `i`.

**Outputs:** 
- `y_i`: The resulting attention output at position i, a weighted combination of past value vectors.

**Limitations:** 
- Quadratic time and memory complexity for long sequences.
- Causal constraint limits dependency to past tokens only.

**Advantages:** 
- Captures sequential dependencies up to position `i`.
- Ensures autoregressive behavior, which is crucial for tasks like language modeling.

---

### Equation (3): Linearized Attention Reformulation
**Equation:** 
```plaintext
y_i = (∑_{j=1}^i φ(Q_i^T K_j)V_j) / (∑_{l=1}^i φ(Q_i^T K_l))
= φ(Q_i)^T (∑_{j=1}^i φ(K_j)V_j) / φ(Q_i)^T (∑_{l=1}^i φ(K_l))
```
**Name:** Linearized Attention Reformulation

**Purpose:** 
To reformulate the attention mechanism using a kernel function φ that allows for linear (in sequence length) computation.

**Inputs/Parameters:** 
- `Q_i`, `K_j`, `V_j`, `φ`: Same as before, but now using a kernel function φ such that φ(x, y) = φ(x)φ(y).

**Logical Flow/Sequence:** 
1. Replace the softmax function used in standard attention with a kernel function φ applied to the dot products.
2. Factor the φ function using the property φ(Q_i^T K_j) = φ(Q_i)^T φ(K_j).
3. Compute weighted sums of the value vectors `V_j` scaled by φ-transformed keys.
4. Normalize by a similar sum over φ-transformed keys to get final attention output.

**Outputs:** 
- `y_i`: The attention output at position `i` using linear attention, computed more efficiently.

**Limitations:** 
- Performance might slightly differ from softmax attention due to approximation.
- Choice of kernel function φ is crucial for effectiveness.

**Advantages:** 
- Reduces computational complexity from quadratic to linear in sequence length.
- Allows scaling to longer sequences.

---

### Equation (4): Recurrent Formulation for Linear Attention Memory Update
**Equation:** 
```plaintext
M_t = M_{t-1} + K_t^T V_t
```
**Name:** Linear Attention Memory Update

**Purpose:** 
To update a memory matrix `M` recurrently for linear attention, accumulating past key-value pairs.

**Inputs/Parameters:** 
- `M_{t-1}`: Memory matrix at time step t-1.
- `K_t`, `V_t`: Key and value vectors at time step t.

**Logical Flow/Sequence:** 
1. At each time step `t`, update the memory matrix by adding the outer product of `K_t` and `V_t` to the previous memory.

**Outputs:** 
- `M_t`: Updated memory matrix at time step t.

**Limitations:** 
- Memory matrix grows or accumulates information over time; can become unwieldy for very long sequences.
- Does not include any forgetting mechanism inherently.

**Advantages:** 
- Allows efficient, recurrent updating suitable for linear attention mechanisms.
- Supports parallelization in training and inference under certain conditions.

---

### Equation (5): Recurrent Formulation for Linear Attention Output
**Equation:** 
```plaintext
y_t = Q_t M_t
```
**Name:** Linear Attention Output via Memory

**Purpose:** 
To compute the output at time step `t` using the current query and the accumulated memory matrix.

**Inputs/Parameters:** 
- `Q_t`: Query vector at time step t.
- `M_t`: Memory matrix at time step t.

**Logical Flow/Sequence:** 
1. Multiply the query vector `Q_t` with the memory matrix `M_t` to get the output.

**Outputs:** 
- `y_t`: The attention output at time step t.

**Limitations:** 
- Assumes that the memory matrix `M_t` correctly encapsulates all necessary past information.

**Advantages:** 
- Very fast computation for each new output once `M_t` is updated.
- Integrates seamlessly with the recurrent update from Equation (4).

---

### Equations (6) and (7): General RNN Read and Write Operations
**Equations:** 
```plaintext
M_t = f(M_{t-1}, x_t) // Write Operation
y_t = g(M_t, x_t) // Read Operation
```
**Name:** General RNN Memory Operations

**Purpose:** 
To define the fundamental operations of a recurrent neural network: updating its hidden state (memory) and producing an output.

**Inputs/Parameters:** 
- `M_{t-1}`: Previous memory/hidden state.
- `x_t`: Current input at time t.
- `f(.,.)`: Write/update function that defines how new input and previous state combine.
- `g(.,.)`: Read/output function that defines how to produce output from the current state and input.

**Logical Flow/Sequence:** 
1. **Write Operation:** Compute new memory `M_t` based on previous memory and current input.
2. **Read Operation:** Compute output `y_t` based on updated memory and current input.

**Outputs:** 
- `M_t`: New memory state after processing input `x_t`.
- `y_t`: Output vector at time t.

**Limitations:** 
- General formulation; actual behavior depends on the choice of functions `f` and `g`.
- Does not specify specifics for handling long-term dependencies.

**Advantages:** 
- Flexible framework applicable to many types of RNN architectures.
- Separates concerns of memory update and output generation.

---

### Equation (8): Surprise-Based Memory Update
**Equation:** 
```plaintext
M_t = M_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)
```
**Name:** Surprise-Based Memory Update

**Purpose:** 
To update the long-term memory by adjusting it in the direction opposite to the gradient of a loss function, scaled by a factor related to "surprise."

**Inputs/Parameters:** 
- `M_{t-1}`: Previous memory state.
- `x_t`: Input at time t.
- `ℓ(M_{t-1}; x_t)`: Loss function evaluating the error of the memory with respect to the current input.
- `∇ℓ(M_{t-1}; x_t)`: Gradient of the loss with respect to the memory.
- `θ_t`: Data-dependent learning rate (step size) at time t.

**Logical Flow/Sequence:** 
1. Compute the gradient of the loss function with respect to the current memory and input.
2. Scale the gradient by learning rate `θ_t`.
3. Subtract the scaled gradient from the previous memory to obtain updated memory.

**Outputs:** 
- `M_t`: Updated memory state at time t.

**Limitations:** 
- Relies on proper choice of loss function ℓ and learning rate θ_t for stability.
- Sensitive to the gradient's quality; poor gradients may lead to ineffective updates.

**Advantages:** 
- Incorporates a notion of “surprise” – larger gradients indicate surprising information, prompting larger updates.
- Directly ties memory updates to prediction errors.

---

### Equation (9): Memory Update with Surprise Term
**Equation:** 
```plaintext
M_t = M_{t-1} + S_t
```
**Name:** Memory Update with Surprise Addition

**Purpose:** 
To update the memory by adding a "surprise" term, encapsulating new information or novelty.

**Inputs/Parameters:** 
- `M_{t-1}`: Previous memory state.
- `S_t`: Surprise metric at time t, representing a change based on new input.

**Logical Flow/Sequence:** 
1. Take the previous memory state.
2. Add the surprise term `S_t` to incorporate new surprising information.

**Outputs:** 
- `M_t`: Updated memory that reflects the addition of surprising new data.

**Limitations:** 
- The effectiveness depends on how well `S_t` captures meaningful changes.

**Advantages:** 
- Separates the concept of surprise from the raw gradient update for clarity and modularity.
- Allows for flexible incorporation of various surprise metrics.

---

### Equation (10): Momentum and Momentary Surprise Update
**Equation:** 
```plaintext
S_t = η_t S_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)
```
**Name:** Momentum and Momentary Surprise Calculation

**Purpose:** 
To update the surprise metric `S_t` by combining a decayed past surprise and the immediate surprise from the current input.

**Inputs/Parameters:** 
- `S_{t-1}`: Previous surprise value.
- `η_t`: Data-dependent surprise decay factor at time t.
- `θ_t`: Learning rate scaling for current gradient.
- `∇ℓ(M_{t-1}; x_t)`: Gradient of loss with respect to the memory given current input.

**Logical Flow/Sequence:** 
1. Decay the past surprise value by factor `η_t`.
2. Compute current momentary surprise as the gradient scaled by `θ_t`.
3. Combine these to form the new surprise metric `S_t`.

**Outputs:** 
- `S_t`: Updated surprise metric reflecting both past context and immediate novelty.

**Limitations:** 
- Choice of decay factor η_t and learning rate θ_t is crucial.
- Might overemphasize recent events if not tuned properly.

**Advantages:** 
- Introduces momentum into surprise measurement, smoothing changes over time.
- Balances historical and current surprise for better memory updates.

---

### Equation (11): Key-Value Projection for Associative Memory
**Equation:** 
```plaintext
k_t = x_t W_K, v_t = x_t W_V
```
**Name:** Key-Value Projection for Associative Memory

**Purpose:** 
To transform the current input `x_t` into a key vector `k_t` and a value vector `v_t` for use in associative memory.

**Inputs/Parameters:** 
- `x_t`: Input at time t.
- `W_K`, `W_V`: Learnable weight matrices for key and value projections.

**Logical Flow/Sequence:** 
1. Multiply `x_t` by `W_K` to obtain key `k_t`.
2. Multiply `x_t` by `W_V` to obtain value `v_t`.

**Outputs:** 
- `k_t`, `v_t`: Projected key and value vectors.

**Limitations:** 
- Standard linear projection limitations; does not capture non-linear relationships by itself.

**Advantages:** 
- Simple way to obtain key-value pairs for memory operations.

---

### Equation (12): Associative Memory Loss
**Equation:** 
```plaintext
ℓ(M_{t-1}; x_t) = ||M_{t-1}(k_t) - v_t||_2^2
```
**Name:** Associative Memory Loss

**Purpose:** 
To quantify the error between the memory module's output when given key `k_t` and the expected value `v_t`.

**Inputs/Parameters:** 
- `M_{t-1}`: Memory module at time t-1.
- `k_t`: Key vector from current input.
- `v_t`: Value vector from current input.

**Logical Flow/Sequence:** 
1. Apply the memory function `M_{t-1}` to `k_t` to obtain a prediction.
2. Compute the squared Euclidean distance between this prediction and `v_t`.

**Outputs:** 
- `ℓ`: Scalar loss value measuring how well the memory recalls `v_t` from `k_t`.

**Limitations:** 
- Assumes a squared loss is appropriate; different tasks might require other losses.

**Advantages:** 
- Provides a clear objective for training the memory module.

---

### Equation (13): Memory Update with Forgetting Mechanism
**Equation:** 
```plaintext
M_t = (1 - α_t)M_{t-1} + S_t
```
**Name:** Memory Update with Forgetting Mechanism

**Purpose:** 
To update the memory while optionally forgetting past content, controlled by a gating parameter.

**Inputs/Parameters:** 
- `M_{t-1}`: Previous memory state.
- `S_t`: Surprise term at time t.
- `α_t`: Forgetting gate parameter at time t (between 0 and 1).

**Logical Flow/Sequence:** 
1. Scale the previous memory by `(1 - α_t)` to forget some part of it.
2. Add the new surprise term `S_t`.
3. Obtain updated memory `M_t`.

**Outputs:** 
- `M_t`: New memory state with forgetting applied.

**Limitations:** 
- Choosing the correct schedule or value for α_t is crucial for balancing memory retention and forgetting.

**Advantages:** 
- Introduces controlled forgetting, preventing memory overflow.

---

### Equation (14): Surprise Update with Forgetting
**Equation:** 
```plaintext
S_t = η_t S_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)
```
**Name:** Surprise Update with Forgetting

**Purpose:** 
Remains the same as Equation (10); restated in the context of the forgetting mechanism, without change.

**Inputs/Parameters:** 
- Same as Equation (10).

**Logical Flow/Sequence:** 
- See Equation (10).

**Outputs:** 
- `S_t`: Surprise metric used in the forgetting-enabled update.

**Limitations & Advantages:** 
- Same as Equation (10).

---

### Equation (15): Memory Retrieval
**Equation:** 
```plaintext
y_t = M^*(q_t)
```
**Name:** Memory Retrieval

**Purpose:** 
To retrieve information from the memory module using a query vector without updating the weights.

**Inputs/Parameters:** 
- `q_t`: Query vector at time t, typically derived from input `x_t`.
- `M^*`: The forward (inference) pass of the memory module without weight updates.

**Logical Flow/Sequence:** 
1. Project input into a query vector `q_t`.
2. Pass `q_t` through the memory module in inference mode to get output `y_t`.

**Outputs:** 
- `y_t`: Retrieved memory output corresponding to the query.

**Limitations:** 
- Quality depends on how well `M^*` has learned to store information.

**Advantages:** 
- Fast retrieval without further training or updates.

---

### Equation (16): Reformulated Memory Update for Parallelization
**Equation:** 
```plaintext
M_t = (1 - α_t)M_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t) = β_tM_0 - ∑_{i=1}^t \frac{θ_iβ_t}{β_i} ∇ℓ(M_{t'}; x_i)
```
**Name:** Reformulated Memory Update for Parallelization

**Purpose:** 
To express the recurrent memory update in a form that is amenable to parallel computation over chunks, incorporating mini-batch gradient descent with weight decay.

**Inputs/Parameters:** 
- `α_t`, `θ_t`: Forgetting and learning rate parameters at time t.
- `β_i`: Product term for weight decay, where \(β_i = \prod_{j=1}^i (1-α_j)\).
- `M_0`: Initial memory.
- Sequence index parameters and chunking indices `t'`, `b`.

**Logical Flow/Sequence:** 
1. The left part restates the update with forgetting and gradient steps.
2. The right-hand side reformulates the update as a sum over past steps weighted by decay factors.
3. This form allows grouping computations across chunks of the sequence for parallel processing.

**Outputs:** 
- `M_t`: Updated memory at time t computed in a parallelizable manner.

**Limitations:** 
- Complexity in managing chunk boundaries and decay factors.
- Depends on proper setting of chunk size `b`.

**Advantages:** 
- Enables parallel computation, improving training efficiency for long sequences.

---

### Equation (17): Mini-Batch Gradient Descent Reformulation
**Equation:** 
```plaintext
∑_{i=1}^b (θ_i/β^b)β_i ∇ℓ(W_0; x_i) = Θ^bB^b(W_0X - X)X^T
```
**Name:** Mini-Batch Gradient Descent Reformulation

**Purpose:** 
To compactly express the gradient computation over a mini-batch during training using matrix operations.

**Inputs/Parameters:** 
- `b`: Chunk size for mini-batch.
- `W_0`: Initial weight matrix of the linear memory module.
- `X`: Matrix containing input tokens for the batch.
- `θ_i`, `β_i`: Learning rate and weight decay parameters for each time step in the batch.
- `Θ^b`, `B^b`: Diagonal matrices constructed from parameters over the batch.

**Logical Flow/Sequence:** 
1. Compute the gradient for each sample in the batch.
2. Weight and sum these gradients using the provided decay and learning rate factors.
3. Express the sum as a product of matrices to leverage efficient linear algebra operations.

**Outputs:** 
- Matrix product yielding the accumulated gradient for the mini-batch.

**Limitations:** 
- Assumes linear memory for simplicity.
- Specific to mini-batch size `b`.

**Advantages:** 
- Converts sequential gradient updates into matrix operations for speed and parallelism.

---

### Equation (18): Momentum Update with Gradient Input
**Equation:** 
```plaintext
S_t = η_t S_{t-1} - θ_t u_t
```
**Name:** Momentum Update with Gradient Input

**Purpose:** 
To update the momentum term `S_t` using the previous momentum and a new gradient input.

**Inputs/Parameters:** 
- `S_{t-1}`: Previous momentum (surprise) state.
- `η_t`: Decay factor for momentum.
- `θ_t`: Learning rate factor.
- `u_t`: Gradient-like input at time t (e.g., ∇ℓ(M_{t'}; x_t)).

**Logical Flow/Sequence:** 
1. Decay the previous momentum by factor `η_t`.
2. Subtract scaled gradient input to update momentum.

**Outputs:** 
- `S_t`: Updated momentum term.

**Limitations:** 
- Similar to earlier momentum limitations regarding parameter tuning.

**Advantages:** 
- Smooths updates over time, leveraging momentum for stability and efficiency.

---

### Equation (19): Augmented Input with Persistent Memory
**Equation:** 
```plaintext
x_new = [p_1, p_2, ..., p_{N_p}] || x
```
**Name:** Augmented Input with Persistent Memory

**Purpose:** 
To prepend a sequence of learnable persistent memory tokens to the input sequence.

**Inputs/Parameters:** 
- `p_1, p_2, ..., p_{N_p}`: Learnable persistent memory embeddings.
- `x`: Original input sequence.
- `||`: Concatenation operator.

**Logical Flow/Sequence:** 
1. Concatenate persistent tokens `[p_1, ..., p_{N_p}]` to the beginning of the input sequence `x`.

**Outputs:** 
- `x_new`: Modified input sequence starting with persistent tokens.

**Limitations:** 
- Increased input length due to added tokens.
- Requires learning appropriate persistent tokens.

**Advantages:** 
- Allows model to incorporate fixed task-specific knowledge from the start of the sequence.

---

### Equation (20): Data-Independent Attention-like Feedforward
**Equation:** 
```plaintext
FFN(x) = W_V Softmax(W_K x)
```
**Name:** Data-Independent Feedforward as Attention

**Purpose:** 
To illustrate that a fully connected layer followed by Softmax can mimic attention weights when parameters are input-independent.

**Inputs/Parameters:** 
- `x`: Input vector.
- `W_K`, `W_V`: Learnable weight matrices.

**Logical Flow/Sequence:** 
1. Project `x` using `W_K`.
2. Apply Softmax to get normalized weights.
3. Multiply by `W_V` to produce the output.

**Outputs:** 
- Resulting vector from the feedforward network.

**Limitations:** 
- Does not adapt weights based on input content dynamically.

**Advantages:** 
- Shows connection between feedforward layers and attention mechanisms.

---

### Equation (21): Retrieving Historical Information
**Equation:** 
```plaintext
h_t = M^*_{t-1}(q_t)
```
**Name:** Retrieve Historical Information

**Purpose:** 
To extract relevant past information from long-term memory using a query vector.

**Inputs/Parameters:** 
- `M^*_{t-1}`: Inference-mode memory module at time t-1.
- `q_t`: Query vector derived from current segment or input.

**Logical Flow/Sequence:** 
1. Feed the query `q_t` into the memory module to retrieve `h_t`.

**Outputs:** 
- `h_t`: Retrieved historical context or memory.

**Limitations:** 
- Quality depends on memory's learned associations.

**Advantages:** 
- Allows the model to incorporate long-term context into current processing.

---

### Equation (22): Concatenated Sequence for Enhanced Attention
**Equation:** 
```plaintext
\tilde{S}(t) = [p_1, p_2, ..., p_{N_p}] || h_t || S(t)
```
**Name:** Concatenated Input for Attention with Memory

**Purpose:** 
To form an input segment that includes persistent memory tokens, retrieved historical information, and the current segment.

**Inputs/Parameters:** 
- `[p_1, ..., p_{N_p}]`: Persistent memory tokens.
- `h_t`: Retrieved historical information at time t.
- `S(t)`: Current segment of the sequence.

**Logical Flow/Sequence:** 
1. Concatenate persistent memory, historical memory, and current segment into one sequence.

**Outputs:** 
- `\tilde{S}(t)`: Enhanced sequence input for attention.

**Limitations:** 
- Sequence length increases, affecting computation.

**Advantages:** 
- Enriches the context for the attention mechanism, incorporating long-term memory.

---

### Equation (23): Attention over Enhanced Sequence
**Equation:** 
```plaintext
y_t = Attn(\tilde{S}(t))
```
**Name:** Attention on Enhanced Sequence

**Purpose:** 
To apply the attention mechanism to the enriched sequence that contains persistent memory, historical context, and current data.

**Inputs/Parameters:** 
- `\tilde{S}(t)`: Enhanced sequence from Equation (22).
- `Attn`: Attention mechanism (could be full, causal, or sliding window attention).

**Logical Flow/Sequence:** 
1. Compute attention over the input `\tilde{S}(t)` to produce output `y_t`.

**Outputs:** 
- `y_t`: Attention output at time t.

**Limitations:** 
- Computational cost increases with sequence length.

**Advantages:** 
- Integrates multiple memory sources into the attention process.

---

### Equation (24): Updating Long-Term Memory with Attention Output
**Equation:** 
```plaintext
M_t = M_{t-1}(y_t)
```
**Name:** Long-Term Memory Update with Attention

**Purpose:** 
To update the long-term memory using the output from the attention mechanism.

**Inputs/Parameters:** 
- `M_{t-1}`: Previous memory state.
- `y_t`: Output from attention at time t.

**Logical Flow/Sequence:** 
1. Use output `y_t` as input to update the memory module, resulting in `M_t`.

**Outputs:** 
- `M_t`: Updated memory state incorporating information from the attention output.

**Limitations:** 
- Details of how `M_{t-1}` processes `y_t` are abstracted.

**Advantages:** 
- Allows dynamic updating of memory based on attention results.

---

### Equation (25): Final Output Combining Attention and Memory Retrieval
**Equation:** 
```plaintext
o_t = y_t ⊗ M^*_t(y_t)
```
**Name:** Final Output Combination

**Purpose:** 
To combine the immediate output of the attention mechanism with additional information retrieved from the updated memory.

**Inputs/Parameters:** 
- `y_t`: Output from attention.
- `M^*_t(y_t)`: Retrieved information from memory using `y_t` as a query.
- `⊗`: A gating or combination operation (could be element-wise multiplication, addition with a non-linear function, etc.).

**Logical Flow/Sequence:** 
1. Retrieve additional context from memory using `y_t`.
2. Combine `y_t` and the retrieved memory output using the gating operation to produce final output `o_t`.

**Outputs:** 
- `o_t`: Final output at time t.

**Limitations:** 
- Choice of gating mechanism affects performance.
- Complexity in combining outputs.

**Advantages:** 
- Merges short-term and long-term information for richer representations.

---

### Equation (26): Augmented Input for Gated Memory Variant
**Equation:** 
```plaintext
\tilde{x} = [p_1, p_2, ..., p_{N_p}] || x
```
**Name:** Augmented Input with Persistent Tokens (Gated Memory Variant)

**Purpose:** 
To prepend persistent memory tokens to the input in the context of the Gated Memory architecture.

**Inputs/Parameters:** 
- `[p_1, ..., p_{N_p}]`: Persistent memory tokens.
- `x`: Original input sequence.

**Logical Flow/Sequence:** 
1. Concatenate persistent tokens and input sequence.

**Outputs:** 
- `\tilde{x}`: Modified input sequence for further processing.

**Limitations & Advantages:** 
- Same as Equation (19); tailored for a specific architectural variant.

---

### Equation (27): Sliding Window Attention with Prefix
**Equation:** 
```plaintext
y = SW-Attn^*(\tilde{x})
```
**Name:** Sliding Window Attention with Prefix

**Purpose:** 
To apply a sliding window version of the attention mechanism on the augmented input sequence.

**Inputs/Parameters:** 
- `\tilde{x}`: Input sequence with persistent memory tokens prepended.
- `SW-Attn^*`: Sliding window attention mechanism that handles sequences piecewise with a fixed-size window, and considers the prefix tokens.

**Logical Flow/Sequence:** 
1. Apply sliding window attention to process `\tilde{x}`, yielding output `y`.

**Outputs:** 
- `y`: Output from the sliding window attention mechanism.

**Limitations:** 
- Sliding window limits the context to a fixed size.

**Advantages:** 
- More efficient for very long sequences than full attention.
- Incorporates prefix information from persistent tokens.

---

### Equation (28): Gating Combination in MAG Architecture
**Equation:** 
```plaintext
o = y ⊗ M(\tilde{x})
```
**Name:** Gated Combination of Attention and Memory

**Purpose:** 
To combine the output of sliding window attention and the output of the memory module using a gating mechanism in the Memory-as-Gate (MAG) architecture.

**Inputs/Parameters:** 
- `y`: Output from sliding window attention.
- `M(\tilde{x})`: Output of the memory module when given augmented input `\tilde{x}`.
- `⊗`: Gating operation (e.g., elementwise multiplication after normalization and non-linearity).

**Logical Flow/Sequence:** 
1. Compute outputs `y` and `M(\tilde{x})`.
2. Apply gating operation to combine them into final output `o`.

**Outputs:** 
- `o`: Combined output incorporating both attention and memory insights.

**Limitations:** 
- Effectiveness depends on chosen gating mechanism.
- Adds complexity to the model architecture.

**Advantages:** 
- Allows dynamic blending of short-term attention and long-term memory outputs.

---

### Equation (29): Augmented Input for MAL Architecture
**Equation:** 
```plaintext
\tilde{x} = [p_1, p_2, ..., p_{N_p}] || x
```
**Name:** Augmented Input with Persistent Tokens (MAL Variant)

**Purpose:** 
Similar to Equation (26), but specific to the Memory-as-a-Layer (MAL) architecture.

**Inputs/Parameters:** 
- `[p_1, ..., p_{N_p}]`: Persistent memory tokens.
- `x`: Original input sequence.

**Logical Flow/Sequence:** 
1. Concatenate persistent memory tokens and input sequence.

**Outputs:** 
- `\tilde{x}`: Modified input sequence for the MAL variant.

**Limitations & Advantages:** 
- Same as previous persistent memory concatenations.

---

### Equation (30): Passing Augmented Input through Neural Memory Layer
**Equation:** 
```plaintext
y = M(\tilde{x})
```
**Name:** Memory Layer Processing

**Purpose:** 
To process the augmented input sequence through the neural memory module, now acting as a layer in the MAL architecture.

**Inputs/Parameters:** 
- `\tilde{x}`: Input sequence with persistent tokens.
- `M`: Neural memory module acting as a layer.

**Logical Flow/Sequence:** 
1. Feed `\tilde{x}` into the memory module to compute an output `y`.

**Outputs:** 
- `y`: Output from the memory layer.

**Limitations:** 
- Purely sequential processing of the entire input if not combined with attention.

**Advantages:** 
- Directly integrates long-term memory computation as part of the network stack.

---

### Equation (31): Applying Sliding Window Attention after Memory Layer
**Equation:** 
```plaintext
o = SW-Attn(y)
```
**Name:** Sliding Window Attention Post-Memory

**Purpose:** 
To apply sliding window attention to the output of the memory layer, integrating local context processing.

**Inputs/Parameters:** 
- `y`: Output from the memory layer.
- `SW-Attn`: Sliding window attention mechanism.

**Logical Flow/Sequence:** 
1. Process `y` with sliding window attention to compute final output `o`.

**Outputs:** 
- `o`: Final output after attending to memory layer outputs.

**Limitations:** 
- Still constrained by sliding window context limitations.

**Advantages:** 
- Combines benefits of long-term memory with efficient local attention.

---

### Equation (32): Linear Memory Update with Forget Gate (Specialized)
**Equation:** 
```plaintext
M_t = diag(1 - α_t) M_{t-1} + S_t
```
**Name:** Linear Memory Update with Forget Gate

**Purpose:** 
To update a linear memory module using element-wise forgetting via a diagonal matrix.

**Inputs/Parameters:** 
- `M_{t-1}`: Previous memory matrix.
- `S_t`: Surprise term at time t.
- `α_t`: Forget gate parameter (vector) at time t.
- `diag(1 - α_t)`: Diagonal matrix where each diagonal element is `1 - α_t` for corresponding dimensions.

**Logical Flow/Sequence:** 
1. Multiply each element of `M_{t-1}` by corresponding `(1 - α_t)` factor.
2. Add the surprise term `S_t`.
3. Update memory to `M_t`.

**Outputs:** 
- `M_t`: New memory state.

**Limitations:** 
- Applies forgetting uniformly across certain dimensions as specified by `α_t`.

**Advantages:** 
- Fine-grained control over memory retention per dimension.

---

### Equation (33): Specialized Update for \(S_t\) in Linear Case
**Equation:** 
```plaintext
S_t = diag(η_t) S_{t-1} - diag(θ_t) (M_{t-1}k_t - v_t) k_t^T
```
**Name:** Linear Surprise Update Specialized

**Purpose:** 
To update the surprise term `S_t` for a linear memory model, incorporating forgetting and linear projections.

**Inputs/Parameters:** 
- `S_{t-1}`: Previous surprise term.
- `η_t`: Decay factor vector at time t.
- `θ_t`: Learning rate vector at time t.
- `M_{t-1}`: Previous memory matrix.
- `k_t`: Current key vector.
- `v_t`: Current value vector.
- `diag(η_t)`, `diag(θ_t)`: Diagonal matrices formed from vectors `η_t` and `θ_t`.

**Logical Flow/Sequence:** 
1. Decay the previous surprise term by multiplying with `diag(η_t)`.
2. Compute error term `(M_{t-1}k_t - v_t)`.
3. Scale and project the error using `diag(θ_t)` and outer product with `k_t^T`.
4. Subtract this from the decayed surprise to get new `S_t`.

**Outputs:** 
- `S_t`: Updated surprise metric for the linear case.

**Limitations:** 
- Specific to linear memory models.

**Advantages:** 
- Provides clear component-wise update with explicit forgetting and learning rates.

---

### Equation (34): Gated DeltaNet Update Rule
**Equation:** 
```plaintext
S_{t+1} = S_t (I - θ_t k_t k_t^T) + θ_t v_t k_t^T
```
**Name:** Gated DeltaNet Update Rule

**Purpose:** 
To update the memory term following the DeltaNet algorithm, incorporating gating and forgetting implicitly.

**Inputs/Parameters:** 
- `S_t`: Current state in DeltaNet update.
- `θ_t`: Learning rate at time t.
- `k_t`, `v_t`: Current key and value vectors.
- `I`: Identity matrix.

**Logical Flow/Sequence:** 
1. Adjust `S_t` by multiplying with a factor that removes influence of past data scaled by `θ_t k_t k_t^T`.
2. Add the term `θ_t v_t k_t^T` to incorporate new information.

**Outputs:** 
- `S_{t+1}`: Updated state according to DeltaNet.

**Limitations:** 
- Assumes linearity and specific gating structure from DeltaNet.

**Advantages:** 
- Proven method from DeltaNet for effective memory updates with gating.

---

### Equation (35): Longhorn Update Rule (with Implicit Online Learning)
**Equation:** 
```plaintext
S_{t+1} = S_t (I - δ_t k_t k_t^T) + δ_t v_t k_t^T,
```
with 
```plaintext
δ_t = θ_t / (1 + θ_t k_t^T k_t)
```
**Name:** Longhorn Update Rule

**Purpose:** 
To update the memory term following the Longhorn algorithm, which adjusts learning rate adaptively based on input.

**Inputs/Parameters:** 
- `S_t`: Current state.
- `θ_t`: Base learning rate.
- `k_t`, `v_t`: Current key and value vectors.
- `I`: Identity matrix.
- `δ_t`: Adapted learning rate computed from `θ_t` and the norm of `k_t`.

**Logical Flow/Sequence:** 
1. Compute adaptive learning rate `δ_t`.
2. Similar to DeltaNet, adjust state `S_t` using factor `(I - δ_t k_t k_t^T)` and incorporate new information `δ_t v_t k_t^T`.

**Outputs:** 
- `S_{t+1}`: Updated state according to Longhorn.

**Limitations:** 
- Complexity in computing adaptive learning rate may add overhead.

**Advantages:** 
- Adaptive adjustment improves stability and performance in certain tasks.


### Equation (1): Linear Projections for Attention
**Name:** Linear Projections for Attention  
**Purpose:** Compute the Query (Q), Key (K), and Value (V) matrices used in the attention mechanism by projecting the input through learned weight matrices.  
**Inputs/Parameters:**  
- `x`: Input matrix of shape (N, d_in), where N is the sequence length and d_in is the input dimension.  
- `W_Q`, `W_K`, `W_V`: Learnable weight matrices of shape (d_in, d_in).  

**General Logical Flow:**  
1. Multiply the input `x` with weight matrix `W_Q` to obtain `Q`.
2. Multiply the input `x` with weight matrix `W_K` to obtain `K`.
3. Multiply the input `x` with weight matrix `W_V` to obtain `V`.

**Outputs:**  
- `Q`, `K`, `V`: Projected matrices used for computing attention.

**Limitations:**  
- Assumes the input and output dimensions are compatible.
- Purely linear; additional non-linearities may be required for complex tasks.

**Advantages:**  
- Simple and efficient computation.
- Provides the foundational representations for subsequent attention mechanisms.

---

### Equation (2): Causal Attention Output
**Name:** Causal Attention Output  
**Purpose:** Compute the output of an autoregressive (causal) attention mechanism for a given position i.  
**Inputs/Parameters:**  
- `Q_i`: Query vector for position i.
- `K_j`: Key vectors for positions j from 1 to i.
- `V_j`: Value vectors for positions j from 1 to i.
- `d_in`: Dimension of input, used for scaling.  

**General Logical Flow:**  
1. For each j ≤ i, calculate the dot product `Q_i^T K_j`.
2. Scale dot products by `1/√d_in`.
3. Apply the exponential function to scaled dot products.
4. Use these exponentials to weight corresponding value vectors `V_j`.
5. Normalize by the sum of exponentials over all positions from 1 to i.

**Outputs:**  
- `y_i`: The output at position i, combining information from all positions up to i.

**Limitations:**  
- Quadratic complexity in sequence length.
- Only considers past and current tokens (causal).

**Advantages:**  
- Accurately captures dependencies up to the current position in a sequence.
- Ensures autoregressive property required for tasks like language modeling.

---

### Equation (3): Linearized Attention Reformulation
**Name:** Linearized Attention Reformulation  
**Purpose:** Reformulate attention using a kernel function φ to achieve linear complexity with respect to sequence length.  
**Inputs/Parameters:**  
- `Q_i`, `K_j`, `V_j`: As previously defined.
- `φ(·)`: Kernel function such that φ(x, y) = φ(x)φ(y).  

**General Logical Flow:**  
1. Replace softmax with kernel function φ applied to dot products.
2. Use factorization property: φ(Q_i^T K_j) = φ(Q_i)^T φ(K_j).
3. Compute weighted sums of values using φ-transformed keys and queries.
4. Normalize by sums over φ(K) terms.

**Outputs:**  
- `y_i`: Attention output at position i computed using linear attention.

**Limitations:**  
- Choice of kernel φ affects performance and approximation fidelity.
- Might trade off some accuracy for efficiency.

**Advantages:**  
- Reduces computational complexity, enabling handling of longer sequences.
- Retains key properties of attention while using cheaper operations.

---

### Equation (4): Recurrent Formulation for Linear Attention Memory Update
**Name:** Linear Attention Memory Update  
**Purpose:** Update a memory matrix incrementally for linear attention mechanisms.  
**Inputs/Parameters:**  
- `M_{t-1}`: Memory matrix from the previous time step.
- `K_t`: Key vector at time t.
- `V_t`: Value vector at time t.

**General Logical Flow:**  
1. Compute the outer product `K_t^T V_t`.
2. Add this product to the previous memory `M_{t-1}` to form `M_t`.

**Outputs:**  
- `M_t`: Updated memory matrix at time t.

**Limitations:**  
- Memory matrix continually grows in information; may require management for very long sequences.
- No forgetting mechanism is included in this equation alone.

**Advantages:**  
- Simple recurrent update compatible with linear attention.
- Facilitates efficient computation and parallelization.

---

### Equation (5): Recurrent Formulation for Linear Attention Output
**Name:** Linear Attention Output via Memory  
**Purpose:** Calculate the attention output at time t using the current query and the accumulated memory.  
**Inputs/Parameters:**  
- `Q_t`: Query vector at time t.
- `M_t`: Memory matrix at time t.

**General Logical Flow:**  
1. Multiply `Q_t` with `M_t`.
2. Produce output vector `y_t`.

**Outputs:**  
- `y_t`: The output at time t.

**Limitations:**  
- Relies on the memory matrix accurately capturing relevant information.

**Advantages:**  
- Very efficient once `M_t` is maintained.
- Integrates seamlessly with the recurrent memory update process.

---

### Equations (6) and (7): General RNN Memory Operations
**Names:**  
- Equation (6): RNN Write Operation  
- Equation (7): RNN Read Operation

**Purposes:**  
- **(6):** Update or write to the memory using current input.  
- **(7):** Read from the memory to produce an output given current input.

**Inputs/Parameters:**  
- `M_{t-1}`: Previous memory/hidden state.
- `x_t`: Current input at time t.
- `f(·, ·)`: Write/update function.
- `g(·, ·)`: Read/output function.

**General Logical Flow:**  
- **Write Operation (6):** Compute `M_t = f(M_{t-1}, x_t)` to update memory with new information.  
- **Read Operation (7):** Compute `y_t = g(M_t, x_t)` to generate output based on updated memory and current input.

**Outputs:**  
- `M_t`: Updated memory state.  
- `y_t`: Output at time t.

**Limitations:**  
- Behavior and capacity depend heavily on definitions of `f` and `g`.
- Does not specify handling of long-term dependencies inherently.

**Advantages:**  
- Provides a flexible abstract framework for any RNN variant.
- Can be specialized for various architectures by choosing different functions f and g.

---

### Equation (8): Surprise-Based Memory Update
**Name:** Surprise-Based Memory Update  
**Purpose:** Update the long-term memory based on the “surprise” from a new input, measured as the gradient of a loss function.  
**Inputs/Parameters:**  
- `M_{t-1}`: Previous memory.
- `x_t`: Current input.
- `ℓ(M_{t-1}; x_t)`: Loss function that quantifies the error at time t given memory.
- `∇ℓ(M_{t-1}; x_t)`: Gradient of the loss with respect to the memory.
- `θ_t`: Learning rate or scaling factor at time t.

**General Logical Flow:**  
1. Calculate the gradient of the loss with respect to the current memory and input.
2. Scale this gradient by `θ_t`.
3. Subtract the scaled gradient from the previous memory to update to `M_t`.

**Outputs:**  
- `M_t`: New memory state incorporating information from the surprising input.

**Limitations:**  
- Requires careful tuning of `θ_t`.
- Heavily dependent on the choice of loss function ℓ.

**Advantages:**  
- Directly ties memory updates to unexpected or novel inputs.
- Can adaptively update memory based on data dynamics.

---

### Equation (9): Memory Update with Surprise Term
**Name:** Memory Update with Surprise Addition  
**Purpose:** Update memory by adding a computed surprise term, rather than using a gradient subtraction directly.  
**Inputs/Parameters:**  
- `M_{t-1}`: Previous memory.
- `S_t`: Surprise term at time t.

**General Logical Flow:**  
1. Add the surprise term `S_t` to the previous memory `M_{t-1}`.

**Outputs:**  
- `M_t`: Updated memory state.

**Limitations:**  
- Effectiveness depends on accurate computation of `S_t`.

**Advantages:**  
- Modular approach: decouples computation of surprise from memory update.
- Simplifies reasoning about what is being added to memory.

---

### Equation (10): Momentum and Momentary Surprise Update
**Name:** Momentum and Momentary Surprise Calculation  
**Purpose:** Update the surprise metric by combining momentum (past surprise) with the immediate surprise from current input.  
**Inputs/Parameters:**  
- `S_{t-1}`: Previous surprise value.
- `η_t`: Data-dependent decay factor at time t.
- `θ_t`: Scaling factor for current gradient.
- `∇ℓ(M_{t-1}; x_t)`: Gradient of the loss function with respect to current memory and input.

**General Logical Flow:**  
1. Decay the previous surprise with factor `η_t`.
2. Compute the immediate surprise as `θ_t ∇ℓ(M_{t-1}; x_t)`.
3. Subtract immediate surprise from decayed previous surprise to produce new surprise `S_t`.

**Outputs:**  
- `S_t`: Updated surprise metric.

**Limitations:**  
- Requires proper tuning of decay and scaling factors.
- May smooth out significant surprises if decay is too high.

**Advantages:**  
- Incorporates both historical and current information, potentially leading to more stable updates.
- Momentum-like behavior can help propagate important signals over time.

---

### Equation (11): Key-Value Projection for Associative Memory
**Name:** Key-Value Projection for Associative Memory  
**Purpose:** Project the input into key and value vectors for later use in an associative memory update.  
**Inputs/Parameters:**  
- `x_t`: Input at time t.
- `W_K`, `W_V`: Learnable weight matrices.

**General Logical Flow:**  
1. Compute `k_t = x_t W_K`.
2. Compute `v_t = x_t W_V`.

**Outputs:**  
- `k_t`, `v_t`: Key and value vectors for time t.

**Limitations:**  
- Linear projections may not capture all complexities without further non-linear processing.

**Advantages:**  
- Simple and directly usable in memory updates.

---

### Equation (12): Associative Memory Loss
**Name:** Associative Memory Loss  
**Purpose:** Define a loss function that measures how well the memory recalls a value given its key.  
**Inputs/Parameters:**  
- `M_{t-1}`: Memory at time t-1.
- `k_t`: Key vector at time t.
- `v_t`: Value vector at time t.

**General Logical Flow:**  
1. Apply the memory function `M_{t-1}` to key `k_t` to produce a prediction.
2. Compute the squared Euclidean distance between this prediction and `v_t`.

**Outputs:**  
- Scalar loss value `ℓ(M_{t-1}; x_t)`.

**Limitations:**  
- Specific to tasks where Euclidean distance is meaningful.

**Advantages:**  
- Clear objective for training the memory module.

---

### Equation (13): Memory Update with Forgetting Mechanism
**Name:** Memory Update with Forgetting Mechanism  
**Purpose:** Update the memory while incorporating a forgetting mechanism that controls how much of the past to keep.  
**Inputs/Parameters:**  
- `M_{t-1}`: Previous memory.
- `α_t`: Forgetting gate parameter at time t (0 ≤ α_t ≤ 1).
- `S_t`: Surprise term at time t.

**General Logical Flow:**  
1. Scale the old memory by `(1 - α_t)` to "forget" part of it.
2. Add the new surprise `S_t`.
3. Yield updated memory `M_t`.

**Outputs:**  
- `M_t`: New memory state.

**Limitations:**  
- Requires tuning of forgetting parameter `α_t`.
  
**Advantages:**  
- Prevents unbounded growth of memory.
- Allows controlled retention of information.

---

### Equation (14): Surprise Update with Forgetting
**Name:** Surprise Update with Forgetting  
**Purpose:** Reiterate the calculation of the surprise term in the context of a forgetting mechanism.  
**Inputs/Parameters:**  
- Identical to Equation (10).

**General Logical Flow:**  
- Same steps as in Equation (10).

**Outputs:**  
- `S_t`: Surprise term used in memory update.

**Limitations & Advantages:**  
- Same as Equation (10).

---

### Equation (15): Memory Retrieval
**Name:** Memory Retrieval  
**Purpose:** Retrieve information from memory using a query without altering memory parameters.  
**Inputs/Parameters:**  
- `q_t`: Query vector at time t.
- `M^*`: Memory module in inference (non-training) mode.

**General Logical Flow:**  
1. Pass query `q_t` through memory module `M^*`.
2. Obtain output `y_t`.

**Outputs:**  
- `y_t`: Retrieved memory content corresponding to the query.

**Limitations:**  
- Depends on how well memory has been trained.

**Advantages:**  
- Fast lookup without updating weights.
- Simple inference procedure.

---

### Equation (16): Reformulated Memory Update for Parallelization
**Name:** Reformulated Memory Update for Parallelization  
**Purpose:** Express memory update in a form that allows parallel computation using matrix multiplications and sums.  
**Inputs/Parameters:**  
- `M_{t-1}`, `α_t`, `θ_t`, sequence elements `x_i`, and indices for chunking.
- `β_t`, `β_i`: Weight decay factors computed from `α` values.
- `t'`: Start index of current chunk.
- `b`: Chunk size.

**General Logical Flow:**  
1. On the left, show standard update incorporating forgetting and gradient descent.
2. On the right, reformulate the update as a weighted sum over past steps, using precomputed factors for parallelism.
3. This transformation enables computation with matrix multiplications and summations instead of sequential loops.

**Outputs:**  
- `M_t`: Updated memory at time t computed in a parallel-friendly manner.

**Limitations:**  
- Complexity in managing chunk boundaries and computation of decay factors.

**Advantages:**  
- Enables efficient training on hardware accelerators by leveraging parallel operations.

---

### Equation (17): Mini-Batch Gradient Descent Reformulation
**Name:** Mini-Batch Gradient Descent Reformulation  
**Purpose:** Reformulate gradient computations over a mini-batch to use matrix operations for improved efficiency.  
**Inputs/Parameters:**  
- `b`: Batch (chunk) size.
- `W_0`: Initial weights.
- `X`: Matrix of inputs for the mini-batch.
- `θ_i`, `β_i`: Learning and decay factors for each step in batch.
- `Θ^b`, `B^b`: Diagonal matrices constructed from the sequences of θ and β over the batch.

**General Logical Flow:**  
1. Compute gradients for each time step in the batch.
2. Weight and sum these gradients using factors from `Θ^b` and `B^b`.
3. Express the sum as a product of matrices to leverage efficient computation.

**Outputs:**  
- A matrix that represents the accumulated gradient for the mini-batch, ready for use in weight updates.

**Limitations:**  
- Assumes linear memory model.
- Specific to the chosen batch size `b`.

**Advantages:**  
- Converts sequential operations to efficient matrix multiplications.
- Improves parallelization on GPUs/TPUs.

---

### Equation (18): Momentum Update with Gradient Input
**Name:** Momentum Update with Gradient Input  
**Purpose:** Update a momentum term in chunk-wise gradient descent using past momentum and current gradient.  
**Inputs/Parameters:**  
- `S_{t-1}`: Previous momentum term.
- `η_t`: Decay factor for momentum.
- `θ_t`: Scaling factor for gradient.
- `u_t`: Current gradient-like input computed from the loss.

**General Logical Flow:**  
1. Decay previous momentum: multiply `S_{t-1}` by `η_t`.
2. Compute current term `θ_t u_t`.
3. Subtract current term from decayed momentum to produce `S_t`.

**Outputs:**  
- `S_t`: Updated momentum term.

**Limitations:**  
- Requires accurate computation of `u_t` and tuning of `η_t` and `θ_t`.

**Advantages:**  
- Introduces momentum to stabilize and speed up convergence.
- Reduces noise in updates.

---

### Decaying Mechanism for Memory Management
**Name:** Decaying Mechanism for Memory Management  
**Purpose:** Manage memory by decaying or forgetting parts of it based on memory size and data surprise, generalizing forgetting mechanisms in RNNs.  
**Inputs/Parameters:**  
- Proportion of memory size relative to sequence length.
- Amount of data surprise (e.g., gradient magnitude, novelty).

**General Logical Flow:**  
1. Compute a decay rate based on the proportion of memory used and how surprising new data is.
2. Apply the decay rate to forget less relevant parts of the memory.
3. Update memory using mechanisms like those in Equations (13)-(15) with adjusted forgetting parameters.

**Outputs:**  
- Efficient management of memory, maintaining capacity and relevance over long sequences.

**Limitations:**  
- Requires accurate measurement of “surprise” and tuning of decay parameters.
- Might forget important information if not tuned well.

**Advantages:**  
- Prevents memory overflow.
- Adapts dynamically to sequence characteristics for better performance.

---

### Fast and Parallelizable Training Algorithm for Deep Neural Long-Term Memory
**Name:** Fast and Parallelizable Training Algorithm  
**Purpose:** Enable efficient training of deep neural long-term memory by tensorizing mini-batch gradient descent to use matrix multiplication and summation operations.  
**Inputs/Parameters:**  
- Training data in mini-batches.
- Parameters such as learning rates (θ), decay factors (α, η), and weight matrices.

**General Logical Flow:**  
1. Reformulate inner-loop weight updates (as in Equations (16)-(17)) to express them in terms of matrix multiplications and sums.
2. Utilize these reformulations to compute updates in parallel across chunks of the sequence.
3. Apply these updates across mini-batches to train the model efficiently on accelerators.

**Outputs:**  
- Updated weights for the neural long-term memory module using parallel computation.

**Limitations:**  
- Complexity in setting up matrix operations correctly.
- Assumes certain structure (e.g., linearity within chunks) which might limit flexibility.

**Advantages:**  
- Significantly faster training due to parallelism.
- Scales well with sequence length by leveraging hardware acceleration.

---

### Method for Calculating Momentum Term in Chunk-wise Gradient Descent Using Parallel Associative Scan
**Name:** Momentum Calculation with Parallel Associative Scan  
**Purpose:** Compute the momentum terms across chunks of a sequence in parallel, improving efficiency.  
**Inputs/Parameters:**  
- Sequence of gradients or inputs `u_t` for each time step in a chunk.
- Decay factors `η_t`, scaling factors `θ_t`.

**General Logical Flow:**  
1. Represent the momentum update (Equation (18)) as a linear recurrence over the chunk.
2. Use a parallel associative scan algorithm to compute the sequence of momentum terms `S_t` for all t in the chunk simultaneously.

**Outputs:**  
- Sequence of momentum terms `S_t` computed in parallel for a chunk.

**Limitations:**  
- Requires specific hardware or libraries that support parallel associative scan.
- Best suited for chunked computations.

**Advantages:**  
- Reduces sequential dependency, speeding up computation.
- Leverages parallelism within hardware accelerators.

---

### Method for Using Parameters as Functions of Chunks
**Name:** Parameters as Functions of Chunks  
**Purpose:** Simplify the model by making parameters constant within chunks rather than input-dependent at each token, improving efficiency.  
**Inputs/Parameters:**  
- Sequence divided into chunks.
- Parameters `α`, `θ`, `η` set per chunk rather than per token.

**General Logical Flow:**  
1. Instead of computing separate values for `α_t`, `θ_t`, `η_t` at each time step based on `x_t`, assign fixed values for each parameter within an entire chunk.
2. Use these chunk-wise constant parameters in memory update and learning rules across the chunk.

**Outputs:**  
- Simplified computation with parameters constant within chunks.

**Limitations:**  
- Reduced expressiveness since parameters are less adaptive to individual tokens.
- May not capture fine-grained variations in data within a chunk.

**Advantages:**  
- Significantly speeds up computation.
- Simplifies model parameterization, reducing overhead.

---

### Titans Architecture Variants
**Name:** Titans Architecture  
**Purpose:** Family of deep learning models incorporating long-term memory with attention, featuring three hyper-heads: Core, Long-term Memory, Persistent Memory.  
**Variants and Purposes:**  
- **Memory as a Context (MAC):** Treats memory as additional context for current inputs.
- **Gated Memory (MAG):** Combines sliding window attention and neural memory using a gating mechanism.
- **Memory as a Layer (MAL):** Uses neural memory as a layer in the network.

**Inputs/Parameters:**  
- Input sequence, persistent tokens, memory module, attention mechanism variants.

**General Logical Flow/Sequence:**  
- **MAC:**  
  1. Retrieve historical context from long-term memory.
  2. Concatenate persistent tokens, retrieved memory, and current segment.
  3. Apply attention to this enhanced sequence.
  4. Update memory based on attention output.
- **MAG:**  
  1. Prepend persistent tokens to input.
  2. Apply sliding window attention.
  3. Update memory directly from input separately.
  4. Combine attention and memory outputs using a gating operation.
- **MAL:**  
  1. Prepend persistent tokens to input.
  2. Pass input through neural memory layer.
  3. Apply sliding window attention to memory layer output.

**Outputs:**  
- Model outputs for each variant, incorporating long-term memory and attention differently.

**Limitations:**  
- Each variant has trade-offs between expressiveness, speed, and ability to handle long contexts.

**Advantages:**  
- Flexibility in designing architectures based on task requirements.
- Leverages memory modules to extend context and performance over long sequences.

---

### Method for Incorporating Persistent Memory into the Model
**Name:** Incorporating Persistent Memory  
**Purpose:** Introduce learnable, input-independent parameters as task-related memory by appending them to the input sequence.  
**Inputs/Parameters:**  
- Persistent memory tokens `[p_1, ..., p_{N_p}]`.
- Original input sequence `x`.

**General Logical Flow:**  
1. Concatenate persistent memory tokens to the beginning of the input sequence.
2. Use this augmented sequence in subsequent processing (attention, memory updates, etc.).

**Outputs:**  
- Augmented input sequence with task-related persistent tokens prepended.

**Limitations:**  
- Increases sequence length.
- Needs training to learn effective persistent tokens.

**Advantages:**  
- Provides static, task-relevant context from the start.
- Can alleviate biases in initial attention layers.

---

### Method for Retrieving Memory without Weight Update
**Name:** Memory Retrieval without Weight Update  
**Purpose:** Retrieve stored information from the memory module for a given query during inference, without modifying weights.  
**Inputs/Parameters:**  
- Query vector `q_t`.
- Trained memory module in inference mode.

**General Logical Flow:**  
1. Compute `y_t = M^*(q_t)` using a forward pass.
2. Do not perform any gradient updates during this retrieval.

**Outputs:**  
- Retrieved memory output `y_t`.

**Limitations:**  
- Retrieval quality tied to how well memory was trained previously.

**Advantages:**  
- Quick lookup without altering model parameters.
- Enables consistent inference behavior.

---

### Use of 1D Depthwise-Separable Convolution after Projections
**Name:** 1D Depthwise-Separable Convolution in Projections  
**Purpose:** Enhance the performance of query, key, and value computations by applying efficient convolutions.  
**Inputs/Parameters:**  
- Projected queries `Q`, keys `K`, and values `V`.
- Convolutional filters and parameters for depthwise-separable convolution.

**General Logical Flow:**  
1. After computing linear projections for Q, K, and V, apply a 1D depthwise-separable convolution layer to each.
2. Process each channel separately then combine channels efficiently.

**Outputs:**  
- Convolved Q, K, V representations with potentially richer local context.

**Limitations:**  
- Slight additional computational overhead.
- Convolution hyperparameters need tuning.

**Advantages:**  
- Can capture local sequential patterns.
- Efficient compared to standard convolutions, improving performance without significant cost.

---

### Use of SiLU Activation and ℓ2-Normalization
**Name:** SiLU Activation with ℓ2-Normalization  
**Purpose:** Use the SiLU (Sigmoid Linear Unit) activation function for computing queries, keys, and values, and normalize them using ℓ2-norm to stabilize training.  
**Inputs/Parameters:**  
- Pre-activation values from linear projections for Q, K, V.

**General Logical Flow:**  
1. Apply SiLU activation to the raw projected values to introduce non-linearity.
2. Normalize the resulting queries and keys using their ℓ2 norm (i.e., divide by the square root of the sum of squares of components).

**Outputs:**  
- Activated and normalized Q and K (and optionally V) vectors.

**Limitations:**  
- Normalization may reduce the range of activations, potentially affecting expressiveness if overapplied.

**Advantages:**  
- SiLU provides smooth, non-linear transformations that can improve model learning.
- ℓ2-normalization stabilizes attention computations by preventing large variation in vector magnitudes.



